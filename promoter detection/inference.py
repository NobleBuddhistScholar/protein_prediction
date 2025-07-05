import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def predict_sequence(text, model_path, kmer=-1):
    """
    使用 Huggingface Transformer 模型对 DNA 序列进行分类预测。

    参数:
        text (str): 待预测的 DNA 序列（如 "ATCGTACG..."）
        model_path (str): 训练好的模型保存目录（需包含 tokenizer 和 model）
        kmer (int): k-mer 切分长度。若模型使用 k-mer（如 GENERator），则设为对应值（如 6）；
                    若模型不使用 k-mer，则设为 -1，表示不传入该参数。

    返回:
        dict: 包含以下四个字段：
            - predicted_index (int): 预测的类别索引（从 0 开始）
            - predicted_label (str): 映射后的类别名称（通过模型 config.id2label 获取）
            - confidence (float): 预测类别的 softmax 置信度，范围为 [0, 1]
            - probabilities (dict): 每个类别名称对应的预测概率（保留四位小数）
    """

    # 构建 tokenizer 的参数字典，启用 trust_remote_code 以支持自定义 tokenizer（如 DNAKmerTokenizer）
    tokenizer_args = {"trust_remote_code": True}
    if kmer != -1:
        tokenizer_args["k"] = kmer  # 若模型需要 k-mer，则传入 k 参数

    # 加载 tokenizer 和分类模型（如 DNABERT、LLAMA 分类模型等）
    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_args)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()  # 设置为推理模式，禁用 dropout 等训练行为

    # 将输入序列编码为模型可接受的张量格式
    inputs = tokenizer(
        text,
        return_tensors="pt",  # 返回 PyTorch Tensor 格式
        truncation=True,  # 超出最大长度时截断
        padding=True,  # 自动填充到同一长度（batch 模型时用）
        max_length=tokenizer.model_max_length  # 最大输入长度，从 tokenizer 配置中获取
    )

    # 模型推理：前向传播并计算 softmax 概率
    with torch.no_grad():  # 禁用梯度计算，节省内存和加速
        outputs = model(**inputs)
        logits = outputs.logits  # 原始输出分数 (未归一化)
        probs = F.softmax(logits, dim=-1).squeeze()  # 对每个类别进行 softmax 概率归一化

        predicted_index = torch.argmax(probs).item()  # 取概率最大的位置作为预测类别
        confidence = probs[predicted_index].item()  # 对应类别的置信度

    # 获取类别索引与标签名的映射表（如 {0: "non-promoter", 1: "promoter"}）
    id2label = model.config.id2label
    predicted_label = id2label[predicted_index]  # 映射得到类别名称

    # 构建完整概率分布（以标签名为 key，概率为 value）
    probabilities = {id2label[i]: round(prob.item(), 4) for i, prob in enumerate(probs)}

    result = {
        "predicted_index": predicted_index,
        "predicted_label": predicted_label,
        "confidence": round(confidence, 4),
        "probabilities": probabilities
    }

    label = "True" if result["predicted_index"] else "False"
    print("预测标签:", label)
    print("预测置信度:", result["confidence"])
    print("完整概率分布:", result["probabilities"])

    # 返回结构化预测结果
    return result


# 测试入口：支持直接运行此脚本进行单句推理
if __name__ == "__main__":
    # 要预测的 DNA 序列（真实使用时可替换）
    text = "CCCTCCCTTATTCGGGGAGAACACTTAGAAACCCGTCCTTGCCGATGCGCTGCATCAGCTCTAGAGGTTG"

    # 调用函数进行预测
    result = predict_sequence(
        text=text,
        model_path="/root/autodl-tmp/MoE/Train_Model/output/GENERator-eukaryote-1.2b-base",
        kmer=6  # 若模型无需 k-mer，可设为 -1
    )

    # 输出预测信息
    print("输入序列：", text)

    # 若模型是二分类，可以将 predicted_index 解释为 True/False
    label = "True" if result["predicted_index"] else "False"
    print("预测标签:", label)
    print("预测置信度:", result["confidence"])
    print("完整概率分布:", result["probabilities"])
