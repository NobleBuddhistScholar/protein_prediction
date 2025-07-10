import os
import csv
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union
import random

import torch
import transformers
import sklearn
import numpy as np
from torch.utils.data import Dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    save_strategy: str = field(default='no')
    eval_steps: int = field(default=100)
    eval_strategy: str = field(default="no"),
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="temp_output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


"""
Get the reversed complement of the original DNA sequence.
"""


def get_alter_of_dna_sequence(sequence: str):
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    # return "".join([MAP[c] for c in reversed(sequence)])
    return "".join([MAP[c] for c in sequence])


"""
Transform a dna sequence to k-mer string
"""


def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i:i + k] for i in range(len(sequence) - k + 1)])


"""
Load or generate k-mer string for each DNA sequence. The generated k-mer string will be saved to the same directory as the original data with the same name but with a suffix of "_{k}mer".
"""


def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """Load or generate k-mer string for each DNA sequence."""
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:
        logging.warning(f"Generating k-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            logging.warning(f"Saving k-mer to {kmer_path}...")
            json.dump(kmer, f)

    return kmer


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
                 data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 kmer: int = -1):

        super(SupervisedDataset, self).__init__()

        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            logging.warning("Perform single sequence classification...")
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
        elif len(data[0]) == 3:
            # data is in the format of [text1, text2, label]
            logging.warning("Perform sequence-pair classification...")
            texts = [[d[0], d[1]] for d in data]
            labels = [int(d[2]) for d in data]
        else:
            raise ValueError("Data format not supported.")

        if kmer != -1:
            # only write file on the first process
            if torch.distributed.is_initialized() and torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()

            logging.warning(f"Using {kmer}-mer as input...")
            texts = load_or_generate_kmer(data_path, texts, kmer)

            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                torch.distributed.barrier()

        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        self.num_labels = len(set(labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.Tensor(labels).long()
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""


def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray):
    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
    }


# from: https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13
def preprocess_logits_for_metrics(logits: Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]

    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])

    return torch.argmax(logits, dim=-1)


"""
Compute metrics used for huggingface trainer.
"""


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return calculate_metric_with_sklearn(predictions, labels)


"""
Set seed 
"""


def setup_seed(seed):
    torch.manual_seed(seed)  # 设置 CPU 上的随机种子
    torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 上的随机种子（多卡）
    np.random.seed(seed)  # 设置 NumPy 的随机种子
    random.seed(seed)  # 设置 Python 内置的随机种子
    torch.backends.cudnn.benchmark = False  # 关闭自动优化，提升可复现性
    torch.backends.cudnn.deterministic = True  # 强制使用确定性算法


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    setup_seed(training_args.seed)
    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    if "nucleotide" in model_args.model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token
    elif "generator" in model_args.model_name_or_path.lower():
        tokenizer.pad_token = tokenizer.eos_token

    # define datasets and data collator
    train_dataset = SupervisedDataset(tokenizer=tokenizer,
                                      data_path=os.path.join(data_args.data_path, "train.csv"),
                                      kmer=data_args.kmer)
    val_dataset = SupervisedDataset(tokenizer=tokenizer,
                                    data_path=os.path.join(data_args.data_path, "dev.csv"),
                                    kmer=data_args.kmer)
    test_dataset = SupervisedDataset(tokenizer=tokenizer,
                                     data_path=os.path.join(data_args.data_path, "test.csv"),
                                     kmer=data_args.kmer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    if "bigbird" in model_args.model_name_or_path.lower():
        print(f"===================loading gena-lm birdbird model:{model_args.model_name_or_path}====================")
        model = transformers.BigBirdForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            num_labels=train_dataset.num_labels,
        )
    elif "generator" in model_args.model_name_or_path.lower():
        print(f"===================loading GENERator model:{model_args.model_name_or_path}====================")
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))
    elif "dna" in model_args.model_name_or_path.lower():
        print(f"===================loading dnabert model:{model_args.model_name_or_path}====================")
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )
    elif "nucleotide" in model_args.model_name_or_path.lower():
        print(
            f"===================loading nucleotide transformer model:{model_args.model_name_or_path}====================")
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )
    elif "lm-bert" in model_args.model_name_or_path.lower():
        print(f"===================loading gena-lm birt model:{model_args.model_name_or_path}====================")
        # 先尝试直接加载AutoModel
        model = transformers.AutoModel.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True
        )

        # 获取模型模块名
        gena_module_name = model.__class__.__module__
        print(f"Model module: {gena_module_name}")

        # 尝试动态加载对应的SequenceClassification类
        import importlib
        try:
            cls = getattr(importlib.import_module(gena_module_name), 'BertForSequenceClassification')

            print(f"Using model class: {cls.__name__}")

            # 重新加载为序列分类模型
            model = cls.from_pretrained(
                model_args.model_name_or_path,
                num_labels=train_dataset.num_labels,
            )
        except AttributeError:
            # 如果找不到对应的类，回退到AutoModelForSequenceClassification
            print("Specific sequence classification class not found, using AutoModelForSequenceClassification")
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                num_labels=train_dataset.num_labels,
            )
    else:
        raise ValueError(f"{model_args.name_or_path} does not exits")

    # configure LoRA
    if model_args.use_lora:
        print(f"===================using LoRA====================")
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=list(model_args.lora_target_modules.split(",")),
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # define trainer
    trainer = transformers.Trainer(model=model,
                                   tokenizer=tokenizer,
                                   args=training_args,
                                   preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                                   compute_metrics=compute_metrics,
                                   train_dataset=train_dataset,
                                   eval_dataset=val_dataset,
                                   data_collator=data_collator)
    trainer.train()

    if training_args.save_model:
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # get the evaluation results from trainer
    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        results = trainer.evaluate(eval_dataset=test_dataset)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "eval_results.json"), "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    train()