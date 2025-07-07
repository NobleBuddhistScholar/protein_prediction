import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from Bio import SeqIO
from Bio.Seq import Seq
from typing import List, Tuple, Dict, Optional
import re
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score, roc_auc_score,f1_score,classification_report
import json
import matplotlib.pyplot as plt
from pre_processing import translate_dna_to_protein,SequenceEncoder,parse_fasta,set_seed
from model import HybridModel,MSA_ResGRUNet,HyperFusionCortex
import os
from promoter_check import predict_promonter

class ProteinClassifier:
    def __init__(self, config: Optional[dict] = None, label_map: Optional[dict] = None,model_info: Optional[dict] = None,train_model = 'HybridModel'):
        self.config = config or {
            'max_length': 5000,
            'batch_size': 32,
            'epochs': 10,
            'learning_rate': 1e-4
        }
        self.label_map = label_map or {
            "NSP1": 0, "NSP10": 1, "NSP11": 2, "NSP12": 3, "NSP13": 4,
            "NSP14": 5, "NSP15": 6, "NSP16": 7, "NSP2": 8, "NSP3": 9,
            "NSP4": 10, "NSP5": 11, "NSP6": 12, "NSP7": 13, "NSP8": 14,
            "NSP9": 15, "envelope_protein": 16, "membrane_protein": 17,
            "nucleocapsid_protein": 18, "spike_protein": 19
        }
        #模型信息
        self.model_info = model_info or {
            "model_type": train_model,
            "data_set":"",
            "evaluate": {
                "epoch": 0.0,
                "train_loss": 0.0,
                "train_accuracy": 0.0,
                "train_auc": 0.0,
                "train_f1_macro": 0.0,
                "val_loss": 0.0,
                "val_accuracy": 0.0,
                "val_auc": 0.0,
                "val_f1_macro": 0.0,
                "learning_rate": self.config['learning_rate']
            }
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

    def train(self, train_data_path: str, save_path: str = "protein_classifier.pth", json_path: str = "protein_classifier.json", val_ratio: float = 0.2, epoch_callback=None):
        """训练模型，支持每个epoch结束后回调/流式输出评估信息"""
        # 更新模型信息
        self.model_info['data_set'] = train_data_path
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
        elif self.train_model == 'HyperFusionCortex':
            self.model = HyperFusionCortex(
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
            # 新增：每个epoch后回调/产出评估信息
            epoch_info = {
                "epoch": epoch + 1,
                "train": train_metrics,
                "val": val_metrics,
                "learning_rate": optimizer.param_groups[0]['lr']
            }
            if epoch_callback:
                epoch_callback(epoch_info)
            # Early stopping
            if val_metrics['accuracy'] > best_acc:
                best_acc = val_metrics['accuracy']
                # 保存最佳模型
                self.model_info['evaluate']['epoch'] = epoch + 1
                self.model_info['evaluate']['train_loss'] = train_metrics['loss']
                self.model_info['evaluate']['train_accuracy'] = train_metrics['accuracy']
                self.model_info['evaluate']['train_auc'] = train_metrics['auc']
                self.model_info['evaluate']['train_f1_macro'] = train_metrics['f1_macro']
                self.model_info['evaluate']['val_loss'] = val_metrics['loss']
                self.model_info['evaluate']['val_accuracy'] = val_metrics['accuracy']
                self.model_info['evaluate']['val_auc'] = val_metrics['auc']
                self.model_info['evaluate']['val_f1_macro'] = val_metrics['f1_macro']
                self.model_info['evaluate']['learning_rate'] = optimizer.param_groups[0]['lr']
                self.save(save_path, json_path)
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
        with open(config_path, 'w') as f:
            json.dump(self.model_info, f, indent=4)

    # 加载预训练模型
    @classmethod
    def per_train_load(cls, model_path: str, config = None,model = 'HybridModel' , model_info_path: str = "protein_classifier.json"):
        """加载预训练模型"""
        # 检查 CUDA 是否可用，如果不可用则将模型加载到 CPU 上
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=device)
        config = config or {
            'max_length': 10000,
            'batch_size': 32,
            'epochs': 10,
            'learning_rate': 1e-4
        }
        classifier = cls(config=config, label_map=checkpoint['label_map'])
        classifier.encoder.max_length = checkpoint['max_length']
        # 加载模型信息
        with open(model_info_path, 'r') as f:
            classifier.model_info = json.load(f)
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
        elif model == 'HyperFusionCortex':
            classifier.model = HyperFusionCortex(
                vocab_size=23,
                embedding_dim=128,
                num_classes=len(checkpoint['label_map'])
            ).to(classifier.device)
            classifier.model.load_state_dict(checkpoint['model_state'])            
            return classifier       

    # 加载注释模型
    @classmethod
    def load(cls, model_path: str, model = 'HybridModel'):
        """加载标注模型"""
        # 检查 CUDA 是否可用，如果不可用则将模型加载到 CPU 上
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=device)
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
        elif model == 'HyperFusionCortex':
            classifier.model = HyperFusionCortex(
                vocab_size=23,
                embedding_dim=128,
                num_classes=len(checkpoint['label_map'])
            ).to(classifier.device)
            classifier.model.load_state_dict(checkpoint['model_state'])            
            return classifier       

    # 实际的基因预测
    def practical_predict_genome(self, fasta_file, result_save_path, min_confidence=0.7, min_protein_length=100, molecule_type='DNA'):
        def promoters_scoring(dna_seq, mol_type):
            """寻找启动子并为启动子打分"""
            # 记录启动子区域以及置信度
            promoters_info = {}
            seq_len = len(dna_seq)

            if mol_type == 'RNA':
                strands = ['+']
                start_codons = {'ATG', 'GTG', 'TTG'}
            else:
                strands = ['+']
                start_codons = {'ATG', 'GTG', 'TTG'}

            for strand in strands:
                working_seq = dna_seq if strand == '+' else str(Seq(dna_seq).reverse_complement())
                seq_len = len(working_seq)
                for frame in range(3):
                    for pos in trange(frame, seq_len - 2, 3, desc=f"Strand {strand} Frame {frame}"):
                        codon = working_seq[pos:pos+3]
                        if codon in start_codons:
                            start_pos = pos
                            # 启动子区域
                            promoter_start = max(0, start_pos - 100)
                            promoter_end = start_pos
                            promoter_region = working_seq[promoter_start:promoter_end]
                            # 计算启动子得分
                            result = predict_promonter(
                                text=promoter_region,
                                model_path="promoter_model/gena-lm-bert-base",
                                kmer=6  # 若模型无需 k-mer，可设为 -1
                            )
                            label = True if result["predicted_index"] else False
                            if label == True:
                                confidence = result["confidence"]
                                promoters_info[start_pos] = confidence
                            else:
                                confidence = 1- result["confidence"]
                                promoters_info[start_pos] = confidence
                        pos += 3
            return promoters_info
                
        # 分子类型敏感的ORF检测
        def advanced_orf_detection(dna_seq, min_len, mol_type, max_length=self.config['max_length']):
            """改进版ORF检测：每个起始密码子可产生多个ORF，支持启动子区域记录"""
            # 先检测启动子信息
            promoters_info = promoters_scoring(dna_seq, mol_type)
            orfs = []
            seq_len = len(dna_seq)
            max_orf_nt = max_length * 3

            if mol_type == 'RNA':
                strands = ['+']
                start_codons = {'ATG', 'GTG', 'TTG'}
            else:
                strands = ['+']
                start_codons = {'ATG', 'GTG', 'TTG'}
            stop_codons = {'TAA', 'TAG', 'TGA'}

            for strand in strands:
                working_seq = dna_seq if strand == '+' else str(Seq(dna_seq).reverse_complement())
                seq_len = len(working_seq)
                for frame in range(3):
                    pos = frame
                    while pos <= seq_len - 3:
                        codon = working_seq[pos:pos+3]
                        if codon in start_codons:
                            start_pos = pos
                            # 向后查找所有终止密码子
                            found_stop = False
                            next_pos = pos + 3
                            orf_count = 0
                            while (next_pos <= min(start_pos + max_orf_nt, seq_len - 3)):
                                next_codon = working_seq[next_pos:next_pos+3]
                                if next_codon in stop_codons:
                                    orf_len = (next_pos + 3 - start_pos) // 3
                                    if orf_len >= min_len:
                                        orfs.append({
                                            'start': start_pos if strand == '+' else (len(dna_seq) - start_pos - 3),
                                            'end': next_pos + 3 if strand == '+' else (len(dna_seq) - next_pos),
                                            'strand': strand,
                                            'length': orf_len,
                                            'stop_quality': 1.0,
                                            'promoter_confidence': promoters_info.get(start_pos, 0.0)
                                        })
                                        orf_count += 1
                                    found_stop = True
                                next_pos += 3
                            # 如果没有终止密码子，或最后一个终止密码子后还有剩余长度
                            if not found_stop or (start_pos + max_orf_nt < seq_len):
                                orf_len = (min(start_pos + max_orf_nt, seq_len) - start_pos) // 3
                                if orf_len >= min_len:
                                    orfs.append({
                                        'start': start_pos if strand == '+' else (len(dna_seq) - start_pos - 3),
                                        'end': min(start_pos + max_orf_nt, seq_len) if strand == '+' else 0,
                                        'strand': strand,
                                        'length': orf_len,
                                        'stop_quality': 0.8,
                                        'promoter_confidence': promoters_info.get(start_pos, 0.0)
                                    })
                        pos += 3
            return sorted(orfs, key=lambda x: (-x['length'], -x['stop_quality']))

        # 多维度预测合并策略
        def intelligent_merge(predictions):
            # 按照置信度和蛋白质长度进行排序
            predictions.sort(key=lambda x: (-x['confidence'], -x['protein_length']))
            
            merged = {}  # 存储每个蛋白质类型的最终结果
            
            for pred in predictions:
                protein_type = pred['prediction']
                # 如果该蛋白质类型之前未添加，则添加当前预测
                if protein_type not in merged:
                    merged[protein_type] = pred
                # 如果已经有该蛋白质的预测且当前预测置信度更高，则更新该蛋白质的预测
                elif pred['confidence'] > merged[protein_type]['confidence']:
                    merged[protein_type] = pred

            # 将最终的预测结果排序
            final = list(merged.values())
            final.sort(key=lambda x: x['start'])
            
            return final

        # 主处理流程
        os.makedirs(result_save_path, exist_ok=True)
        for record in SeqIO.parse(fasta_file, "fasta"):
            dna_seq = str(record.seq).upper()
            print(dna_seq)
            orfs = advanced_orf_detection(dna_seq, min_protein_length // 3, molecule_type)
            
            predictions = []
            for orf in orfs:
                try:
                    if orf['strand'] == '+':
                        fragment = dna_seq[orf['start']:orf['end']]
                        promoter_confidence = orf.get('promoter_confidence', 0.0)
                    else:
                        fragment = str(Seq(dna_seq).reverse_complement()[orf['start']:orf['end']])
                        promoter_confidence = orf.get('promoter_confidence', 0.0)
                    
                    if molecule_type == 'RNA' and orf['strand'] == '+':
                        fragment = fragment.replace('T', 'U')  # 转换为RNA序列
                        promoter_confidence = orf.get('promoter_confidence', 0.0)
                    
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
                    # 蛋白质分类
                    with torch.no_grad():
                        outputs = self.model(tensor)
                        probs = torch.softmax(outputs, dim=1)
                        protein_conf, pred_idx = torch.max(probs, 1)

                    # 综合置信度
                    conf = (protein_conf.item() + promoter_confidence) / 2
                    if conf >= min_confidence:
                        predictions.append({
                            'start': orf['start'] + 1,
                            'end': orf['end'],
                            'strand': orf['strand'],
                            'prediction': self.inverse_label_map[pred_idx.item()],
                            'confidence': conf,
                            'protein_length': len(protein),
                            'stop_quality': orf.get('stop_quality', 1.0)
                        })
                except Exception as e:
                    logging.error(f"ORF处理异常: {str(e)}")
            
            final_preds = intelligent_merge(predictions)
            
            # self.visualize_genome(final_preds, len(dna_seq), f"{save_path}/{record.id}_map.png")
            
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
                            "protein_length": p['protein_length']
                        }
                    } for p in final_preds
                ]
            }
            with open(f"{result_save_path}/{record.id}_annotation.json", "w") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            
        return output

    # 用于演示的基因预测
    def predict_genome(self, fasta_file, result_save_path, min_confidence=0.7, min_protein_length=100, molecule_type='DNA'):
        # 分子类型敏感的ORF检测
        def advanced_orf_detection(dna_seq, min_len, mol_type):
            """支持RNA/DNA双模式的ORF检测，遍历所有超过最短序列限制的ORF"""
            orfs = []
            seq_len = len(dna_seq)

            # 确定链方向与起始密码子
            if mol_type == 'RNA':
                strands = ['+']  # 默认RNA只处理正链
                start_codons = {'ATG', 'GTG', 'TTG'}  # RNA序列起始密码子
            else:
                strands = ['+', '-']  # 同时处理正链与反链
                start_codons = {'ATG', 'GTG', 'TTG'}  # DNA序列起始密码子
            
            stop_codons = {'TAA', 'TAG', 'TGA'}  # 终止密码子

            # 遍历正链和反链
            for strand in strands:
                # 正链或反链的DNA序列
                working_seq = dna_seq if strand == '+' else str(Seq(dna_seq).reverse_complement())
                
                for frame in range(3):  # 三个阅读框架
                    state_map = {}  # ORF状态跟踪

                    for pos in range(frame, len(working_seq)-2, 3):  # 从frame开始，以步长3遍历序列
                        codon = working_seq[pos:pos+3]
                        
                        # 检测起始密码子
                        if codon in start_codons:
                            genome_pos = pos if strand == '+' else (len(dna_seq) - pos - 3)
                            state_map[genome_pos] = {
                                'start': genome_pos,
                                'end': None,
                                'length': 0
                            }
                        
                        expired = []  # 过期的ORF
                        for start in list(state_map.keys()):
                            current_orf = state_map[start]
                            
                            # 如果是终止密码子，判断是否结束
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
                                # 继续增加ORF长度
                                current_orf['length'] += 1
                        
                        # 清理过期的ORF
                        for start in expired:
                            del state_map[start]

                    # 检查未结束的ORF
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

        # 多维度预测合并策略
        def intelligent_merge(predictions):
            # 按照置信度和蛋白质长度进行排序
            predictions.sort(key=lambda x: (-x['confidence'], -x['protein_length']))
            
            merged = {}  # 存储每个蛋白质类型的最终结果
            used_stop_codons = set()  # 记录已使用的终止密码子位置
            protein_types_added = set()  # 记录已添加的蛋白质类型
            
            for pred in predictions:
                protein_type = pred['prediction']
                stop_codon_pos = pred['end']  # 终止密码子的位置
                
                # 终止密码子位置必须不同于已经使用的
                if stop_codon_pos in used_stop_codons:
                    continue
                
                # 如果该蛋白质类型之前未添加，则添加当前预测
                if protein_type not in protein_types_added:
                    merged[protein_type] = pred
                    protein_types_added.add(protein_type)
                    used_stop_codons.add(stop_codon_pos)  # 将当前蛋白的终止密码子位置标记为已使用
                
                # 如果已经有该蛋白质的预测且当前预测置信度更高，则更新该蛋白质的预测
                elif pred['confidence'] > merged[protein_type]['confidence']:
                    # 需要检查终止密码子位置是否冲突
                    if stop_codon_pos not in used_stop_codons:
                        merged[protein_type] = pred
                        used_stop_codons.add(stop_codon_pos)  # 更新终止密码子位置为已使用

            # 将最终的预测结果排序
            final = list(merged.values())
            final.sort(key=lambda x: x['start'])
            
            return final

        # 主处理流程
        os.makedirs(result_save_path, exist_ok=True)
        for record in SeqIO.parse(fasta_file, "fasta"):
            dna_seq = str(record.seq).upper()
            print(dna_seq)
            orfs = advanced_orf_detection(dna_seq, min_protein_length // 3, molecule_type)
            
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
            
            # self.visualize_genome(final_preds, len(dna_seq), f"{save_path}/{record.id}_map.png")
            
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
                            "protein_length": p['protein_length']
                        }
                    } for p in final_preds
                ]
            }
            with open(f"{result_save_path}/{record.id}_annotation.json", "w") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            
        return output

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
if __name__ == "__main__":
    # 测试模型标注
    classifier = ProteinClassifier.load(model_path='model/HybridModel_v1.pth', model='HybridModel')
    genome_result = classifier.predict_genome(
        "prectice_test/TS000000.fasta", 
        result_save_path="test_results",
        min_confidence=0.7,
        min_protein_length=100,
        molecule_type='DNA'
    )
    print(json.dumps(genome_result, indent=2, ensure_ascii=False))

    # # 测试模型训练
    # config = {
    #     'max_length': 10000,
    #     'batch_size': 32,
    #     'epochs': 2,
    #     'learning_rate': 1e-4
    # }
    # classifier = ProteinClassifier.per_train_load(model_path='model/HybridModel_v1.pth',config=config,model_info_path='model/HybridModel_v1.json', model='HybridModel')
    # classifier.train(
    #     train_data_path="train_data_tt",
    #     save_path="HybridModel_v2.pth",
    #     json_path="HybridModel_v2.json",
    #     val_ratio=0.2
    # )