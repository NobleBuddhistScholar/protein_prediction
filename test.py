import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from Bio import SeqIO
from Bio.Seq import Seq
from typing import List, Tuple, Dict, Optional
import re
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score,f1_score,classification_report
import json
import matplotlib.pyplot as plt
from transformers import BertModel, BertConfig
import json
import logging
import random
# ============================
# 配置日志
# ============================
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

# ============================
# 数据编码模块
# ============================
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

# ============================
# PyTorch模型定义
# ============================
class HybridModel(nn.Module):
    """CNN-BiLSTM混合模型"""
    def __init__(self, vocab_size: int, embedding_dim: int, num_classes: int):
        super().__init__()
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # CNN部分
        self.conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(2)
        
        # BiLSTM部分
        self.lstm1 = nn.LSTM(256, 256, bidirectional=True, batch_first=True)
        self.dropout1 = nn.Dropout(0.5)
        
        self.lstm2 = nn.LSTM(512, 128, bidirectional=True, batch_first=True)
        self.dropout2 = nn.Dropout(0.5)
        
        # 全连接层
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # 添加层初始化
        self._init_weights()
    
    def _init_weights(self):
        for layer in [self.fc1, self.fc2]:
            nn.init.kaiming_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0.1)
        
    def forward(self, x):
        # 输入形状: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # CNN处理
        x = x.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.pool1(x)  # (batch_size, 128, seq_len//2)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.pool2(x)  # (batch_size, 256, seq_len//4)
        
        # LSTM处理
        x = x.permute(0, 2, 1)  # (batch_size, seq_len//4, 256)
        x, _ = self.lstm1(x)    # (batch_size, seq_len//4, 512)
        x = self.dropout1(x)
        
        # 获取最后一个时间步的输出
        _, (h_n, _) = self.lstm2(x)  # h_n: (2, batch_size, 128)
        x = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch_size, 256)
        x = self.dropout2(x)
        
        # 全连接层
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x

class MSA_ResGRUNet(nn.Module):
    def __init__(self, vocab_size=5, embedding_dim=64, num_classes=2):
        super().__init__()
        # 嵌入层（考虑padding_idx=0）
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 多尺度卷积模块
        self.multi_scale = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embedding_dim, 64, k, padding=k//2),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=3, stride=2)  # 长度减半
            ) for k in [3, 5, 7]
        ])
        
        # 残差卷积块
        self.res_blocks = nn.Sequential(
            ResidualBlock(192, 256, dilation=1),
            ResidualBlock(256, 256, dilation=2),
            nn.MaxPool1d(kernel_size=3, stride=2)  # 长度再减半
        )
        
        # 自适应压缩层
        self.adaptive_compress = nn.Sequential(
            nn.AdaptiveAvgPool1d(512),  # 动态调整到固定长度
            nn.Conv1d(256, 128, 1),     # 通道压缩
            nn.GELU()
        )
        
        # 高效注意力模块
        self.attention = nn.Sequential(
            LocalAttention(128, window_size=32),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        # 双向GRU
        self.gru = nn.GRU(128, 64, bidirectional=True, batch_first=True, num_layers=2)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # 输入形状: (B, L)
        x = self.embedding(x)  # (B, L, E)
        x = x.permute(0, 2, 1)  # (B, E, L)
        
        # 多尺度特征提取
        features = []
        for module in self.multi_scale:
            features.append(module(x))
        x = torch.cat(features, dim=1)  # (B, 192, L//2)
        
        # 残差卷积处理
        x = self.res_blocks(x)  # (B, 256, L//8)
        
        # 自适应压缩
        x = self.adaptive_compress(x)  # (B, 128, 512)
        x = x.permute(0, 2, 1)  # (B, 512, 128)
        
        # 局部注意力增强
        x = self.attention(x)  # (B, 512, 128)
        
        # 双向GRU
        x, _ = self.gru(x)  # (B, 512, 128)
        
        # 动态池化
        x = F.adaptive_avg_pool1d(x.permute(0, 2, 1), 1)  # (B, 128, 1)
        x = x.squeeze(-1)  # (B, 128)
        
        return self.classifier(x)

class ResidualBlock(nn.Module):
    """带扩张卷积的残差块"""
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, 
                             padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3,
                             padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.gelu(x)

class LocalAttention(nn.Module):
    """高效局部注意力机制"""
    def __init__(self, dim, window_size=31):
        super().__init__()
        self.qkv = nn.Linear(dim, dim*3)
        self.proj = nn.Linear(dim, dim)
        self.window_size = window_size
        self.dim = dim

    def forward(self, x):
        B, L, C = x.shape
        # 动态填充至窗口整数倍
        pad_len = (self.window_size - L % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_len, 0, 0))  # 在序列末尾填充
        
        # 分块计算
        new_L = L + pad_len
        x = x.view(B, new_L // self.window_size, self.window_size, C)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) / (C ** 0.5)
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).view(B, L, C)
        return self.proj(x[:, :L, :])

# ============================
# 分类器主类
# ============================
class ProteinClassifier:
    def __init__(self, config: Optional[dict] = None, label_map: Optional[dict] = None,train_model = 'HybridModel'):
        self.config = config or {
            'max_length': 10000,
            'batch_size': 32,
            'epochs': 50,
            'threshold': 0.5,
            'learning_rate': 1e-4
        }
        self.label_map = label_map or {
            "NSP1": 0, "NSP10": 1, "NSP11": 2, "NSP12": 3, "NSP13": 4,
            "NSP14": 5, "NSP15": 6, "NSP16": 7, "NSP2": 8, "NSP3": 9,
            "NSP4": 10, "NSP5": 11, "NSP6": 12, "NSP7": 13, "NSP8": 14,
            "NSP9": 15, "envelope_protein": 16, "membrane_protein": 17,
            "nucleocapsid_protein": 18, "spike_protein": 19
        }

        self.encoder = SequenceEncoder(self.config['max_length'])
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.inverse_label_map = {v: k for k, v in self.label_map.items()}
        self.train_model = train_model

        set_seed(42)
        
    def _create_label_matrix(self, labels: List[List[str]]) -> np.ndarray:
        """创建多标签矩阵"""
        label_matrix = np.zeros((len(labels), len(self.label_map)), dtype=np.float32)
        for i, lbls in enumerate(labels):
            for label in lbls:
                if label in self.label_map:
                    label_matrix[i, self.label_map[label]] = 1.0
        return label_matrix
    def _create_label_indices(self, labels: List[List[str]]) -> np.ndarray:
        """将多热标签转换为类别索引"""
        return np.argmax(self._create_label_matrix(labels), axis=1)

    def _load_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载并预处理数据"""
        sequences, labels = parse_fasta(data_path)
        translated = [translate_dna_to_protein(seq) for seq in sequences]
        valid_indices = [i for i, seq in enumerate(translated) if len(seq) > 0]
        translated = [translated[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]
        
        X = self.encoder.transform(translated)
        y = self._create_label_matrix(labels)
        return X, y

    def train(self, train_data_path: str,save_path: str = "protein_classifier.pth", val_ratio: float = 0.2):
        """训练模型"""
        # 加载训练数据
        train_sequences, train_labels = parse_fasta(train_data_path)
        translated_train = [translate_dna_to_protein(seq) for seq in train_sequences]
        valid_indices = [i for i, seq in enumerate(translated_train) if len(seq) > 0]
        translated_train = [translated_train[i] for i in valid_indices]
        train_labels = [train_labels[i] for i in valid_indices]
        
        # 编码数据
        self.encoder.fit(translated_train)
        X = self.encoder.transform(translated_train)
        y_indices = self._create_label_indices(train_labels)

        # 划分训练集和验证集
        dataset_size = len(X)
        indices = np.random.permutation(dataset_size)
        split_idx = int(dataset_size * (1 - val_ratio))
        
        # 转换为Tensor
        train_dataset = TensorDataset(
            torch.LongTensor(X[indices[:split_idx]]),
            torch.LongTensor(y_indices[indices[:split_idx]]) 
        )
        val_dataset = TensorDataset(
            torch.LongTensor(X[indices[split_idx:]]),
            torch.LongTensor(y_indices[indices[split_idx:]])
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'],
            pin_memory=True
        )
        
        # 初始化模型
        if self.train_model ==  'HybridModel':
            self.model = HybridModel(
                vocab_size=23,
                embedding_dim=128,
                num_classes=len(self.label_map)
            ).to(self.device)
        elif self.train_model =='MSA_ResGRUNet':
            self.model = MSA_ResGRUNet(
                vocab_size=23,
                embedding_dim=128,
                num_classes=len(self.label_map)
            ).to(self.device)            
                
        # 训练配置
        criterion = nn.CrossEntropyLoss() 
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)
        
        best_acc = 0.0
        early_stop_counter = 0
        patience = 10
        
        # 训练循环
        for epoch in range(self.config['epochs']):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
            
            # 评估阶段
            train_metrics = self._evaluate(train_loader, criterion)
            val_metrics = self._evaluate(val_loader, criterion)
            
            # 更新学习率
            scheduler.step(val_metrics['accuracy'])  # 根据验证集AUC调整学习率
            
            # 打印详细指标
            logging.info(
                f"\nEpoch {epoch+1}/{self.config['epochs']}:"
                f"\n  Train Loss: {train_metrics['loss']:.4f} | "
                f"Acc: {train_metrics['accuracy']:.4f} | "
                f"AUC: {train_metrics['auc']:.4f} | "
                f"F1(macro): {train_metrics['f1_macro']:.4f}"
                f"\n  Val Loss:   {val_metrics['loss']:.4f} | "
                f"Acc: {val_metrics['accuracy']:.4f} | "
                f"AUC: {val_metrics['auc']:.4f} | "
                f"F1(macro): {val_metrics['f1_macro']:.4f}"
                f"\n  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}"
            )
            
            # Early stopping
            if val_metrics['accuracy'] > best_acc:
                best_acc = val_metrics['accuracy']
                self.save(save_path, "config.json")
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    logging.info("Early stopping triggered")
                    break
        

    def predict(self, test_data_path: str, num: int=5):
        # 加载测试数据
        test_sequences, test_labels = parse_fasta(test_data_path)
        translated_test = [translate_dna_to_protein(seq) for seq in test_sequences]
        valid_indices = [i for i, seq in enumerate(translated_test) if len(seq) > 0]
        translated_test = [translated_test[i] for i in valid_indices]
        test_labels = [test_labels[i] for i in valid_indices]
        
        # 编码数据（使用训练时确定的max_length，不应重新fit）
        X = self.encoder.transform(translated_test)
        y_indices = self._create_label_indices(test_labels)
        
        # 转换为TensorDataset
        test_dataset = TensorDataset(
            torch.LongTensor(X),
            torch.LongTensor(y_indices))
        
        # 创建逐个样本加载的DataLoader
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        # 结果存储
        all_true = []
        all_pred = []
        results = []
        
        self.model.eval()
        with torch.no_grad():
            progress_bar = tqdm(enumerate(test_loader), 
                                total=len(test_loader),
                                desc="Predicting samples",
                                unit="sample")
            
            for idx, (inputs, labels) in progress_bar:
                # 转移到设备
                inputs = inputs.to(self.device)
                
                # 预测
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                # 转换标签
                true_idx = labels.item()
                pred_idx = preds.item()
                true_label = self.inverse_label_map[true_idx]
                pred_label = self.inverse_label_map[pred_idx]
                
                # 收集结果
                all_true.append(true_idx)
                all_pred.append(pred_idx)
                results.append({
                    "sequence": test_sequences[valid_indices[idx]],
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "correct": true_label == pred_label
                })
        
        # 打印详细预测结果
        print("\nIndividual Predictions:")
        for i, res in enumerate(results[:num]):  # 打印前5个示例
            print(f"Sample {i+1}:")
            print(f"Sequence: {res['sequence'][:50]}...")
            print(f"True: {res['true_label']} | Pred: {res['pred_label']}")
            print(f"Status: {'CORRECT' if res['correct'] else 'WRONG'}\n")
        
        # 计算准确率
        accuracy = accuracy_score(all_true, all_pred)
        print(f"\nFinal Accuracy: {accuracy:.4f}")
        
        # 返回详细结果和统计
        return {
            "accuracy": accuracy,
            "total_samples": len(results),
            "correct_count": sum(1 for r in results if r['correct']),
            "detailed_results": results
        }
        
    def _evaluate(self, data_loader: DataLoader, criterion) -> Tuple[float, float, float, float]:
        """改进的评估函数"""
        '''
        Accuracy	正确预测数 / 总样本数	0.0-1.0	整体分类准确率
        AUC (macro)	对每个类别计算AUC后取平均	0.5-1.0	模型区分正负类的能力
        F1 (macro)	每个类别的F1值取平均	0.0-1.0	平衡精确率和召回率
        F1 (micro)	全局统计TP/FP/FN后计算	0.0-1.0	考虑样本不均衡的F1值
        Classification Report	展示每个类别的精确率、召回率、F1值	-	识别表现不佳的具体类别
        '''
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                
                # 获取预测结果
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(probs, 1)
                
                all_probs.append(probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # 合并结果
        all_probs = np.concatenate(all_probs)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        # 计算指标
        metrics = {
            'loss': total_loss / len(data_loader.dataset),
            'accuracy': accuracy_score(all_labels, all_preds),
            'auc': roc_auc_score(
                all_labels, all_probs, 
                multi_class='ovr', 
                average='macro'
            ),
            'f1_macro': f1_score(all_labels, all_preds, average='macro'),
            'f1_micro': f1_score(all_labels, all_preds, average='micro')
        }
        
        # 打印分类报告
        print("\nClassification Report:")
        print(classification_report(
            all_labels, all_preds,
            target_names=list(self.label_map.keys()),
            digits=4
        ))
        
        return metrics 
    
    def save(self, model_path: str, config_path: str):
        """保存模型和配置"""
        torch.save({
            'model_state': self.model.state_dict(),
            'config': self.config,
            'label_map': self.label_map,
            'max_length': self.encoder.max_length
        }, model_path)
        
    @classmethod
    def load(cls, model_path: str,model = 'HybridModel'):
        """加载预训练模型"""
        checkpoint = torch.load(model_path)
        classifier = cls(config=checkpoint['config'], label_map=checkpoint['label_map'])
        classifier.encoder.max_length = checkpoint['max_length']
        if model == 'HybridModel':
            classifier.model = HybridModel(
                vocab_size=23,
                embedding_dim=128,
                num_classes=len(checkpoint['label_map'])
            ).to(classifier.device)
            classifier.model.load_state_dict(checkpoint['model_state'])
            return classifier
        elif model == 'MSA_ResGRUNet':
            classifier.model = MSA_ResGRUNet(
                vocab_size=23,
                embedding_dim=128,
                num_classes=len(checkpoint['label_map'])
            ).to(classifier.device)
            classifier.model.load_state_dict(checkpoint['model_state'])
            return classifier            


    def predict_genome(self, fasta_file, save_path, min_confidence=0.7, min_protein_length=100, molecule_type='DNA',summary = True):
        # 优化1：分子类型敏感的ORF检测
        def advanced_orf_detection(dna_seq, min_len, mol_type):
            """支持RNA/DNA双模式的ORF检测"""
            orfs = []
            seq_len = len(dna_seq)
            if mol_type == 'RNA':
                strands = ['+']  # 默认RNA只处理正链
                start_codons = {'ATG'}
            else:
                strands = ['+', '-']  # DNA处理双链
                start_codons = {'ATG', 'GTG', 'TTG'}
            
            stop_codons = {'TAA', 'TAG', 'TGA'}
            
            for strand in strands:
                working_seq = dna_seq if strand == '+' else str(Seq(dna_seq).reverse_complement())
                
                for frame in range(3):
                    state_map = {}  # ORF状态跟踪
                    for pos in range(frame, len(working_seq)-2, 3):
                        codon = working_seq[pos:pos+3]

                        if codon in start_codons:
                            genome_pos = pos if strand == '+' else (len(dna_seq) - pos - 3)
                            state_map[genome_pos] = {
                                'start': genome_pos,
                                'end': None,
                                'length': 0
                            }

                        expired = []
                        for start in list(state_map.keys()):
                            current_orf = state_map[start]
                            if codon in stop_codons:
                                orf_len = (pos + 3 - current_orf['start']) // 3
                                if orf_len >= min_len:
                                    orfs.append({
                                        'start': current_orf['start'],
                                        'end': pos + 3,
                                        'strand': strand,
                                        'length': orf_len,
                                        'stop_quality': 1.0
                                    })
                                expired.append(start)
                            else:
                                current_orf['length'] += 1

                        for start in expired:
                            del state_map[start]

                    for start, orf in state_map.items():
                        if orf['length'] >= min_len * 1.5:
                            orfs.append({
                                'start': start,
                                'end': len(working_seq),
                                'strand': strand,
                                'length': orf['length'],
                                'stop_quality': 0.6
                            })

            return sorted(orfs, key=lambda x: (-x['length'], -x['stop_quality']))

        # 优化2：多维度预测合并策略
        def intelligent_merge(predictions):
            predictions.sort(key=lambda x: (-x['confidence'], -x['protein_length']))
            merged = []
            for pred in predictions:
                conflict = False
                for existing in merged:
                    overlap = max(0, min(pred['end'], existing['end']) - max(pred['start'], existing['start']))
                    iou = overlap / (pred['end']-pred['start'] + existing['end']-existing['start'] - overlap)
                    
                    if iou > 0.3:
                        if pred['prediction'] == existing['prediction']:
                            existing['start'] = min(existing['start'], pred['start'])
                            existing['end'] = max(existing['end'], pred['end'])
                            existing['confidence'] = max(existing['confidence'], pred['confidence'])
                            conflict = True
                            break
                        elif pred['confidence'] > existing['confidence'] * 1.5:
                            merged.remove(existing)
                
                if not conflict:
                    merged.append(pred)
            
            final = []
            for pred in merged:
                expected_length = {'spike_protein': (1200, 1400), 'nucleocapsid_protein': (400, 500)}.get(pred['prediction'], (min_protein_length, None))
                if (pred['protein_length'] >= expected_length[0] and (not expected_length[1] or pred['protein_length'] <= expected_length[1])):
                    final.append(pred)
            
            return sorted(final, key=lambda x: x['start'])

        # 主处理流程
        os.makedirs(save_path, exist_ok=True)
        for record in SeqIO.parse(fasta_file, "fasta"):
            dna_seq = str(record.seq).upper()
            orfs = advanced_orf_detection(dna_seq, min_protein_length//3, molecule_type)
            
            predictions = []
            for orf in orfs:
                try:
                    if orf['strand'] == '+':
                        fragment = dna_seq[orf['start']:orf['end']]
                    else:
                        fragment = str(Seq(dna_seq).reverse_complement()[orf['start']:orf['end']])
                    
                    if molecule_type == 'RNA' and orf['strand'] == '+':
                        fragment = fragment.replace('T', 'U')  # 转换为RNA序列
                    
                    protein = []
                    for i in range(0, len(fragment)-2, 3):
                        codon = fragment[i:i+3]
                        aa = str(Seq(codon).translate(table=1))[0]  # 使用标准密码子表
                        if aa == '*': break
                        protein.append(aa)
                    protein = ''.join(protein)
                    
                    if len(protein) < min_protein_length:
                        continue
                    
                    encoded_seq = protein.ljust(self.config['max_length'], 'X')[:self.config['max_length']]
                    encoded = self.encoder.transform([encoded_seq])
                    
                    tensor = torch.LongTensor(encoded).to(self.device)
                    with torch.no_grad():
                        outputs = self.model(tensor)
                        probs = torch.softmax(outputs, dim=1)
                        conf, pred_idx = torch.max(probs, 1)
                    
                    if conf.item() >= min_confidence:
                        predictions.append({
                            'start': orf['start'] + 1,
                            'end': orf['end'],
                            'strand': orf['strand'],
                            'prediction': self.inverse_label_map[pred_idx.item()],
                            'confidence': conf.item(),
                            'protein_length': len(protein),
                            'stop_quality': orf.get('stop_quality', 1.0)
                        })
                except Exception as e:
                    logging.error(f"ORF处理异常: {str(e)}")
            
            final_preds = intelligent_merge(predictions)
            
            self.visualize_genome(final_preds, len(dna_seq), f"{save_path}/{record.id}_map.png")
            
            output = {
                "metadata": {
                    "genome_id": record.id,
                    "length": len(dna_seq),
                    "detection_parameters": {
                        "min_confidence": min_confidence,
                        "min_length": min_protein_length,
                        "model_version": self.model.__class__.__name__
                    }
                },
                "features": [
                    {
                        "type": p['prediction'],
                        "location": f"{p['start']}..{p['end']}",
                        "strand": p['strand'],
                        "qualifiers": {
                            "confidence": round(p['confidence'], 3),
                            "protein_length": p['protein_length'],
                            "partial": p['stop_quality'] < 0.9
                        }
                    } for p in final_preds
                ]
            }
            # 返回数据给 API 处理
            if summary == True:
                response = generate_summary(record.id, output,save_path='result')
            with open(f"{save_path}/{record.id}_annotation.json", "w") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            
        return output,response

    def visualize_genome(self, predictions, seq_length, save_path):
        """增强的可视化方法"""
        plt.figure(figsize=(25, 5))
        ax = plt.gca()

        cmap = plt.get_cmap('tab20')
        color_map = {name: cmap(i/len(self.label_map)) 
                    for i, name in enumerate(self.label_map)}

        for idx, pred in enumerate(predictions):
            color = color_map[pred['prediction']]
            start, end = pred['start'], pred['end']
            strand = pred['strand']

            ax.add_patch(plt.Rectangle(
                (start, 0.3), end-start, 0.4,
                facecolor=color, alpha=0.6,
                edgecolor='black', lw=0.5
            ))

            arrow_length = min(100, (end-start)/3)
            if strand == '+':
                arrow_start = end - arrow_length
                ax.arrow(arrow_start, 0.5, arrow_length-20, 0, 
                        head_width=0.2, head_length=20, fc=color, ec='black')
            else:
                arrow_start = start + arrow_length
                ax.arrow(arrow_start, 0.5, -arrow_length+20, 0, 
                        head_width=0.2, head_length=20, fc=color, ec='black')

            label_x = (start + end) / 2
            ax.text(label_x, 0.7, pred['prediction'],
                ha='center', va='bottom', fontsize=8, rotation=45)
            ax.text(label_x, 0.25, f"{pred['confidence']:.2f}",
                ha='center', va='top', fontsize=6)

        ax.set_xlim(0, seq_length)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Genomic Position (bp)", fontsize=10)
        ax.set_title("Genome Annotation Result", fontsize=12)

        scale_length = 1000
        ax.plot([100, 100+scale_length], [0.9, 0.9], color='black', lw=1)
        ax.text(100+scale_length/2, 0.85, f"{scale_length} bp",
            ha='center', va='top', fontsize=8)

        legend_patches = [plt.Line2D([0], [0], color=color, lw=4, label=name)
                        for name, color in color_map.items()]
        ax.legend(handles=legend_patches, 
                bbox_to_anchor=(1.05, 1), 
                loc='upper left',
                title="Protein Types",
                fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

import json
import logging
from typing import Optional
from openai import OpenAI
# 初始化客户端 - 从环境变量获取API密钥
client = OpenAI(
    api_key='sk-2197cbc777064fc6b7fff4ac5c81959d',  # 请设置环境变量DEEPSEEK_API_KEY
    base_url="https://api.deepseek.com"  # 使用基础URL
)

def generate_summary(genome_id: str, genome_data: dict, save_path: str) -> Optional[str]:
    try:
        # 1. 准备对话消息
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的生物信息学助手，擅长分析病毒基因组序列和蛋白质功能预测。"
                           "请用专业但易懂的语言进行解释，并给出有依据的推测。"
            },
            {
                "role": "user",
                "content": f"""请分析以下基因组预测结果：
                
                **基因组基本信息**
                - 编号: {genome_id}
                - 长度: {genome_data['metadata']['length']} bp
                
                **预测的开放阅读框(ORFs)**:
                {json.dumps(genome_data['features'], indent=2, ensure_ascii=False)}
                
                请从以下方面进行专业分析：
                1. 整体基因组结构特征
                2. 各预测蛋白的功能分析
                3. 可能的病毒分类及生物学意义
                4. 特别关注关键蛋白的特征"""
            }
        ]

        # 2. 调用DeepSeek API
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,  # 控制创造性
            max_tokens=2000,
            top_p=0.9,       # 增加生成多样性
            stream=False
        )

        # 3. 提取生成的总结
        summary = response.choices[0].message.content.strip()
        
        if not summary:
            logging.error("API返回空响应")
            return None

        # 4. 保存结果
        os.makedirs(save_path, exist_ok=True)
        summary_path = os.path.join(save_path, f"{genome_id}_summary.txt")
        
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        
        logging.info(f"总结已保存到: {summary_path}")
        return summary

    except Exception as e:
        logging.error(f"生成总结时出错: {str(e)}", exc_info=True)
        return None
      
# ============================
# 主程序
# ============================
if __name__ == "__main__":
    config =  {
            'max_length': 10000,
            'batch_size': 32,
            'epochs': 50,
            'threshold': 0.5,
            'learning_rate': 1e-4
        }
    label_map = {
            "NSP1": 0, "NSP10": 1, "NSP11": 2, "NSP12": 3, "NSP13": 4,
            "NSP14": 5, "NSP15": 6, "NSP16": 7, "NSP2": 8, "NSP3": 9,
            "NSP4": 10, "NSP5": 11, "NSP6": 12, "NSP7": 13, "NSP8": 14,
            "NSP9": 15, "envelope_protein": 16, "membrane_protein": 17,
            "nucleocapsid_protein": 18, "spike_protein": 19
        }
    # # 初始化分类器
    # classifier = ProteinClassifier(config=config,label_map=label_map,train_model='MSA_ResGRUNet')
    
    # # 训练模型
    # classifier.train("train_data",save_path='model/MSA_ResGRUNet_protein_classifier.pth', val_ratio=0.2)
    
    
    # 加载模型进行预测
    classifier = ProteinClassifier.load("model/MSA_ResGRUNet_protein_classifier.pth",model='MSA_ResGRUNet')
    # # 进行预测
    # results = classifier.predict("test_data")

    # 处理病毒基因组
    results = classifier.predict_genome(
        "prectice_test/sequences.fasta",
        "result",
        min_confidence=0.9,
        molecule_type= 'RNA',
        min_protein_length=50,
        summary = True
    )
    

