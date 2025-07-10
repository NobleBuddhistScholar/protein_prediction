import os
import logging
from pathlib import Path

# === 严格使用您提供的变量名和值 ===
UPLOAD_FOLDER = "prectice_test"       # 注意是prectice_test不是practice_test
RESULT_FOLDER = "result"
SUMMARY_FOLDER = "summary"
REPORTS_FOLDER = "summary"            # 与SUMMARY_FOLDER相同
KNOWLEDGE_FOLDER = "knowledge_files"  # 注意是knowledge_files
GFF_FOLDER = "gff"
ALLOWED_EXTENSIONS = {'fasta'}

# === 日志配置 ===
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def init_folders():
    """在instance目录下创建所有指定文件夹"""
    base_dir = Path(__file__).parent / "instance"
    folders = {
        "UPLOAD_FOLDER": UPLOAD_FOLDER,
        "RESULT_FOLDER": RESULT_FOLDER,
        "SUMMARY_FOLDER": SUMMARY_FOLDER,
        "KNOWLEDGE_FOLDER": KNOWLEDGE_FOLDER,
        "GFF_FOLDER": GFF_FOLDER
    }
    
    for var_name, folder_name in folders.items():
        folder_path = base_dir / folder_name
        if not folder_path.exists():
            os.makedirs(folder_path)
            logging.debug(f"Created folder: {folder_path}")
        
        # 将路径更新为绝对路径
        globals()[var_name] = str(folder_path)