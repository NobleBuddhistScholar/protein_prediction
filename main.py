from train import ProteinClassifier
import torch

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
    results,summary = classifier.predict_genome(
        "prectice_test/sequences.fasta",
        "result",
        'summary',
        min_confidence=0.9,
        molecule_type= 'RNA',
        min_protein_length=50,
        summary = True
    )

    