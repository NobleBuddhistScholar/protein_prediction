import os
import logging
import numpy as np
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from typing import List, Tuple, Dict, Optional
import re
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score,f1_score,classification_report
import json
import logging
import random
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

setup_logging()

def set_seed(seed=42):
    """设置随机种子，确保结果可重复"""
    torch.manual_seed(seed)  # 设置CPU的随机种子
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子
    np.random.seed(seed)  # 设置numpy的随机种子
    random.seed(seed)  # 设置Python内建的随机模块的种子
    torch.backends.cudnn.deterministic = True  # 设置CUDNN为确定性模式
    torch.backends.cudnn.benchmark = False  # 禁用CUDNN的自动优化

# ============================
# 数据预处理模块
# ============================
def parse_fasta(file_path: str) -> Tuple[List[str], List[List[str]]]:
    """读取FASTA文件/目录，返回序列和标签列表"""
    sequences = []
    labels = []
    files = []

    try:
        if os.path.isdir(file_path):
            files = [
                os.path.join(file_path, f)
                for f in os.listdir(file_path)
                if f.lower().endswith(('.fasta', '.fa'))
            ]
        else:
            files = [file_path]

        for file in files:
            protein_name = os.path.basename(file).split('.')[0]
            for record in SeqIO.parse(file, "fasta"):
                seq = str(record.seq).upper()
                sequences.append(seq)
                labels.append([protein_name])
                logging.info(f"Loaded: {file} | Label: {protein_name} | Seq: {seq[:30]}...")

    except Exception as e:
        logging.error(f"Error parsing FASTA: {str(e)}")
        raise

    return sequences, labels

def translate_dna_to_protein(dna_sequence: str) -> str:
    """将DNA序列翻译为蛋白质序列"""
    try:
        cleaned = re.sub(r"[^ATCG]", "X", dna_sequence.upper())
        if len(cleaned) % 3 != 0:
            cleaned = cleaned[:-(len(cleaned) % 3)]
        seq = Seq(cleaned)
        protein = str(seq.translate(to_stop=False))
        protein = re.sub(r'\*', 'X', protein)
        logging.info(f"Translated: {dna_sequence[:30]}... -> {protein[:30]}...")
        return protein
    except Exception as e:
        logging.error(f"Translation failed: {str(e)}")
        return ""

class SequenceEncoder:
    """序列编码器，支持动态长度处理"""
    def __init__(self, max_length: Optional[int] = None):
        self.aa_dict = {aa: idx+1 for idx, aa in enumerate("ACDEFGHIKLMNPQRSTVWY*X")}  # 0为padding
        self.max_length = max_length
        
    def fit(self, sequences: List[str]) -> None:
        """自动确定最大序列长度"""
        if self.max_length is None:
            self.max_length = max(len(seq) for seq in sequences) if sequences else 1000
    
    def transform(self, sequences: List[str]) -> np.ndarray:
        """将序列编码为固定长度的整数数组"""
        encoded = np.zeros((len(sequences), self.max_length), dtype=np.int32)
        for i, seq in enumerate(tqdm(sequences, desc="Encoding")):
            processed = list(seq[:self.max_length].ljust(self.max_length, 'X'))
            encoded[i] = [self.aa_dict.get(aa, 0) for aa in processed]
        return encoded