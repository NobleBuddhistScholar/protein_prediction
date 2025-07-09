import json
import os
import logging
import requests
from openai import OpenAI
from datetime import datetime
from flask import Response, stream_with_context
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from train import ProteinClassifier
from generate_gff import generate_gff
import shutil
import base64
from chat_api import generate_summary_stream
from transformers import AutoTokenizer, AutoModel, AutoConfig
from huggingface_hub import snapshot_download
import shutil

# 配置日志记录
logging.basicConfig(level=logging.DEBUG)

# 配置项
UPLOAD_FOLDER = "prectice_test"
#MODEL_PATH = "model/HyperFusionCortex_v1.pth"
RESULT_FOLDER = "result"
SUMMARY_FOLDER = "summary"
# 新增报告文件夹配置
REPORTS_FOLDER = "summary" 
# 知识库文件夹配置
KNOWLEDGE_FOLDER = "knowledge_files" 
ALLOWED_EXTENSIONS = {'fasta'}
# 在配置项部分新增GFF文件夹配置
GFF_FOLDER = "gff"


app = Flask(__name__)
# 明确指定允许的源
CORS(app) 

# 确保上传目录存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)
if not os.path.exists(SUMMARY_FOLDER):
    os.makedirs(SUMMARY_FOLDER)
# 确保报告文件夹存在
if not os.path.exists(REPORTS_FOLDER):
    os.makedirs(REPORTS_FOLDER)
# 确保GFF文件夹存在
if not os.path.exists(GFF_FOLDER):
    os.makedirs(GFF_FOLDER)
# 确保知识库文件夹存在
if not os.path.exists(KNOWLEDGE_FOLDER):
    os.makedirs(KNOWLEDGE_FOLDER)
# ============================
# 通用方法
# ============================
# 新增：获取模型列表的接口
@app.route('/models', methods=['GET'])
def get_models():
    """获取可用模型列表"""
    try:
        model_files = []
        
        # 检查model文件夹是否存在
        if not os.path.exists('model'):
            return jsonify({"error": "Model directory not found"}), 404
        
        # 遍历model文件夹中的.pth文件
        for file in os.listdir('model'):
            if file.endswith('.pth'):
                # 生成友好的显示名称（去掉.pth后缀，替换下划线为空格）
                display_name = file.replace('.pth', '').replace('_', ' ')
                
                # 检查是否有对应的json文件
                json_file = file.replace('.pth', '.json')
                json_path = os.path.join('model', json_file)
                
                model_info = {
                    'filename': file,
                    'display_name': display_name,
                    'description': '',
                    'model_type': ''
                }
                
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            json_data = json.load(f)
                            model_type = json_data.get('model_type', '未知')
                            model_info['model_type'] = model_type
                            model_info['description'] = f"模型类型: {model_type}"
                            
                            # 如果json中有display_name，优先使用
                            if 'display_name' in json_data:
                                model_info['display_name'] = json_data['display_name']
                            
                    except Exception as e:
                        logging.warning(f"Failed to read model json {json_file}: {str(e)}")
                        model_info['description'] = '配置文件读取失败'
                else:
                    model_info['description'] = '缺少配置文件'
                
                model_files.append(model_info)
        
        # 按文件名排序
        model_files.sort(key=lambda x: x['filename'])
        return jsonify({"models": model_files})
        
    except Exception as e:
        logging.error(f"Failed to get models: {str(e)}")
        return jsonify({"error": f"Failed to get models: {str(e)}"}), 500

# 检查文件类型
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ============================
# 基因组注释模块方法
# ============================
#上传文件并处理基因组预测
@app.route('/upload', methods=['POST'])
def upload_file():
    logging.debug("Received file upload request")
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Only .fasta files are allowed."}), 400

    fasta_file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    try:
        file.save(fasta_file_path)
        logging.debug(f"File saved to {fasta_file_path}")
    except Exception as e:
        logging.error(f"Failed to save file: {str(e)}")
        return jsonify({"error": f"Failed to save file: {str(e)}"}), 500

    # 新增：获取模型文件名参数
    model_file = request.form.get('model_file', 'HyperFusionCortex_v1.pth')
    
    # 验证模型文件是否存在
    model_path = os.path.join('model', model_file)
    if not os.path.exists(model_path):
        return jsonify({"error": f"Model file {model_file} not found"}), 400
        
    model_json = model_file.replace('.pth', '.json')
    model_json_path = os.path.join('model', model_json)
    
    # 检查json文件是否存在并读取模型类型
    if not os.path.exists(model_json_path):
        return jsonify({"error": f"Model configuration file {model_json} not found"}), 400
        
    try:
        with open(model_json_path, 'r', encoding='utf-8') as f:
            model_config = json.load(f)
            model_type = model_config.get('model_type', 'default')
            logging.debug(f"Using model: {model_file}, type: {model_type}")
    except Exception as e:
        logging.error(f"Failed to read model json: {str(e)}")
        return jsonify({"error": f"Failed to read model json: {str(e)}"}), 500

    try:
        # 处理病毒基因组
        classifier = ProteinClassifier.load(model_path=model_path, model=model_type)
        logging.debug("Model loaded successfully")      
        results = classifier.predict_genome(
            fasta_file=fasta_file_path,
            result_save_path=RESULT_FOLDER,
            min_confidence=0.8,
            molecule_type='RNA',
            min_protein_length=100,
        )
        # 直接返回json格式的预测结果
        return {"results": results}
    except Exception as e:
        logging.error(f"Failed to process genome: {str(e)}")
        return jsonify({"error": f"Failed to process genome: {str(e)}"}), 500

# 生成报告（流式分析）
@app.route('/stream_summary', methods=['POST'])
def stream_summary():
    data = request.get_json()
    genome_data = data['genome_data']
    genome_id = genome_data['metadata']['genome_id']
    save_path = SUMMARY_FOLDER
    
    # 获取前端传递的是否使用知识库增强的参数
    use_rag = data.get('use_rag', True)  # 默认为True，保持向后兼容
    
    logging.info(f"生成报告 - 知识库增强: {'开启' if use_rag else '关闭'}")
    
    return Response(
        stream_with_context(
            generate_summary_stream(genome_id, genome_data, save_path, if_rag=use_rag)
        ),
        mimetype='text/plain'
    )

# 获取报告
@app.route('/summary', methods=['GET'])
def get_summary():
    genome_id = request.args.get('genome_id')
    current_date = request.args.get('current_date')

    if not genome_id or not current_date:
        return jsonify({"error": "Missing 'genome_id' or 'current_date' parameter"}), 400

    summary_filename = f"{genome_id}_基因组分析报告_{current_date}.txt"
    summary_file_path = os.path.join(SUMMARY_FOLDER, summary_filename)

    if not os.path.exists(summary_file_path):
        return jsonify({"error": f"Summary file {summary_filename} not found"}), 404

    try:
        with open(summary_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            return jsonify({summary_filename: content})
    except Exception as e:
        logging.error(f"Failed to read summary file {summary_filename}: {str(e)}")
        return jsonify({"error": f"Failed to read summary file: {str(e)}"}), 500

# 利用generate_gff.py生成gff报告
@app.route('/generate_gff', methods=['GET'])
def generate_gff_api():
    genome_id = request.args.get('genome_id')
    current_date = request.args.get('current_date')

    if not genome_id or not current_date:
        return jsonify({"error": "Missing 'genome_id' or 'current_date' parameter"}), 400

    json_file_name = f'{genome_id}_annotation.json'
    json_file_path = os.path.join(RESULT_FOLDER, json_file_name)

    if not os.path.exists(json_file_path):
        return jsonify({"error": f"JSON file {json_file_name} not found"}), 404

    gff_file_name = f'{genome_id}_annotation_{current_date}.gff'
    gff_file_path = os.path.join(GFF_FOLDER, gff_file_name)

    try:
        generate_gff(json_file_path, gff_file_path)
        return jsonify({"message": "GFF报告生成成功"})
    except Exception as e:
        logging.error(f"Failed to generate GFF file {gff_file_name}: {str(e)}")
        return jsonify({"error": f"Failed to generate GFF file: {str(e)}"}), 500

# 获取/gff路径下的gff报告
@app.route('/gff', methods=['GET'])
def get_gff():
    genome_id = request.args.get('genome_id')
    current_date = request.args.get('current_date')

    if not genome_id or not current_date:
        return jsonify({"error": "Missing 'genome_id' or 'current_date' parameter"}), 400

    gff_file_name = f'{genome_id}_annotation_{current_date}.gff'
    gff_file_path = os.path.join(GFF_FOLDER, gff_file_name)  # 修改路径为GFF_FOLDER

    if not os.path.exists(gff_file_path):
        return jsonify({"error": f"GFF file {gff_file_name} not found"}), 404

    try:
        with open(gff_file_path, 'r') as f:
            content = f.read()
            return jsonify({gff_file_name: content})
    except Exception as e:
        logging.error(f"Failed to read GFF file {gff_file_name}: {str(e)}")
        return jsonify({"error": f"Failed to read GFF file: {str(e)}"}), 500

# ============================
# 模型管理模块方法
# ============================
@app.route('/models/<model_filename>', methods=['DELETE'])
def delete_model(model_filename):
    """删除指定模型文件"""
    try:
        # 验证文件名
        if not model_filename.endswith('.pth'):
            return jsonify({"error": "Invalid model filename"}), 400
            
        model_path = os.path.join('model', model_filename)
        json_path = os.path.join('model', model_filename.replace('.pth', '.json'))
        
        # 检查文件是否存在
        if not os.path.exists(model_path):
            return jsonify({"error": "Model file not found"}), 404
            
        # 删除模型文件
        os.remove(model_path)
        logging.info(f"Deleted model file: {model_path}")
        
        # 删除对应的json文件（如果存在）
        if os.path.exists(json_path):
            os.remove(json_path)
            logging.info(f"Deleted model config file: {json_path}")
            
        return jsonify({"message": "Model deleted successfully"})
        
    except Exception as e:
        logging.error(f"Failed to delete model {model_filename}: {str(e)}")
        return jsonify({"error": f"Failed to delete model: {str(e)}"}), 500

@app.route('/models/<model_filename>/rename', methods=['PUT'])
def rename_model(model_filename):
    """重命名模型文件"""
    try:
        data = request.get_json()
        new_name = data.get('new_name')
        
        if not new_name:
            return jsonify({"error": "New name is required"}), 400
            
        # 确保新名称以.pth结尾
        if not new_name.endswith('.pth'):
            new_name += '.pth'
            
        # 验证原文件名
        if not model_filename.endswith('.pth'):
            return jsonify({"error": "Invalid model filename"}), 400
            
        old_model_path = os.path.join('model', model_filename)
        new_model_path = os.path.join('model', new_name)
        old_json_path = os.path.join('model', model_filename.replace('.pth', '.json'))
        new_json_path = os.path.join('model', new_name.replace('.pth', '.json'))
        
        # 检查原文件是否存在
        if not os.path.exists(old_model_path):
            return jsonify({"error": "Model file not found"}), 404
            
        # 检查新文件名是否已存在
        if os.path.exists(new_model_path):
            return jsonify({"error": "A model with this name already exists"}), 409
            
        # 重命名模型文件
        os.rename(old_model_path, new_model_path)
        logging.info(f"Renamed model file: {old_model_path} -> {new_model_path}")
        
        # 重命名对应的json文件（如果存在）
        if os.path.exists(old_json_path):
            os.rename(old_json_path, new_json_path)
            logging.info(f"Renamed model config file: {old_json_path} -> {new_json_path}")
            
        return jsonify({"message": "Model renamed successfully", "new_filename": new_name})
        
    except Exception as e:
        logging.error(f"Failed to rename model {model_filename}: {str(e)}")
        return jsonify({"error": f"Failed to rename model: {str(e)}"}), 500

@app.route('/models/<model_filename>/details', methods=['GET'])
def get_model_details(model_filename):
    """获取模型详细信息"""
    try:
        # 验证文件名
        if not model_filename.endswith('.pth'):
            return jsonify({"error": "Invalid model filename"}), 400
            
        model_path = os.path.join('model', model_filename)
        json_path = os.path.join('model', model_filename.replace('.pth', '.json'))
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            return jsonify({"error": "Model file not found"}), 404
            
        # 获取文件基本信息
        file_stats = os.stat(model_path)
        file_size = file_stats.st_size
        file_modified = file_stats.st_mtime
        
        # 格式化文件大小
        if file_size < 1024:
            size_str = f"{file_size} B"
        elif file_size < 1024*1024:
            size_str = f"{file_size/1024:.1f} KB"
        elif file_size < 1024*1024*1024:
            size_str = f"{file_size/(1024*1024):.1f} MB"
        else:
            size_str = f"{file_size/(1024*1024*1024):.1f} GB"
            
        model_details = {
            "filename": model_filename,
            "file_size": size_str,
            "file_size_bytes": file_size,
            "last_modified": file_modified,
            "model_type": "未知",
            "description": "",
            "config_available": False,
            "config_data": {}
        }
        
        # 读取JSON配置文件
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    model_details.update({
                        "model_type": config_data.get('model_type', '未知'),
                        "description": config_data.get('description', ''),
                        "config_available": True,
                        "config_data": config_data
                    })
            except Exception as e:
                logging.warning(f"Failed to read model config {json_path}: {str(e)}")
                model_details["description"] = "配置文件读取失败"
        else:
            model_details["description"] = "缺少配置文件"
            
        return jsonify(model_details)
        
    except Exception as e:
        logging.error(f"Failed to get model details for {model_filename}: {str(e)}")
        return jsonify({"error": f"Failed to get model details: {str(e)}"}), 500
@app.route('/models/<model_filename>/export', methods=['GET'])
def export_model(model_filename):
    """直接返回指定模型文件内容"""
    try:
        if not model_filename.endswith('.pth'):
            return jsonify({"error": "Invalid model filename"}), 400

        model_path = os.path.join('model', model_filename)
        if not os.path.exists(model_path):
            return jsonify({"error": "Model file not found"}), 404

        # 直接返回文件内容，content-type为二进制流
        from flask import send_file
        return send_file(model_path, mimetype='application/octet-stream')
    except Exception as e:
        logging.error(f"Failed to export model {model_filename}: {str(e)}")
        return jsonify({"error": f"Failed to export model: {str(e)}"}), 500

@app.route('/models/pull/promoter', methods=['POST'])
def pull_promoter_model():
    """从Hugging Face拉取启动子识别模型"""
    try:
        # 配置模型下载相关参数
        model_repo = "yipingjushi/gena-lm-bert-base"
        model_dir = "promoter_model"
        model_name = "gena-lm-bert-base"  # 模型名称
        target_dir = os.path.join(model_dir, model_name)
        
        # 检查目录是否已存在
        if os.path.exists(target_dir):
            return jsonify({
                "success": False,
                "message": f"启动子识别模型 '{model_name}' 已存在",
                "error": "模型已存在，如需更新请先删除现有模型"
            }), 400
        
        # 确保目录结构存在
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        logging.info(f"开始从 Hugging Face 下载启动子识别模型: {model_repo}")
        
        # 配置代理和SSL设置
        os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'
        os.environ['CURL_CA_BUNDLE'] = ''  # 禁用证书验证
        
        # 设置系统环境变量代理
        os.environ['http_proxy'] = 'http://127.0.0.1:7890'
        os.environ['https_proxy'] = 'http://127.0.0.1:7890'
        os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
        os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
        
        # 同时配置proxies参数
        proxies = {
            "http": "http://127.0.0.1:7890",
            "https": "http://127.0.0.1:7890"
        }
        
        logging.info(f"正在使用本地代理 (端口7890) 下载模型")
        
        # 使用 huggingface_hub 的 snapshot_download 函数一次性下载整个模型仓库
        from huggingface_hub import snapshot_download
        
        try:
            logging.info("尝试方法1：使用huggingface_hub的snapshot_download")
            download_path = snapshot_download(
                repo_id=model_repo,
                local_dir=target_dir,
                local_dir_use_symlinks=False,  # 不使用符号链接，直接下载文件
                revision="main",  # 使用主分支
                proxies=proxies,  # 使用设置的代理
                resume_download=True,  # 启用断点续传
                force_download=False,  # 避免强制重新下载
                max_workers=1  # 减少并行下载数量，提高稳定性
            )
        except Exception as hub_error:
            logging.warning(f"尝试方法1失败: {str(hub_error)}")
            
            # 方法2：使用transformers的自动下载功能
            try:
                logging.info("尝试方法2：使用transformers的AutoModel下载")
                from transformers import AutoModel, AutoTokenizer, AutoConfig
                
                # 创建目录
                os.makedirs(target_dir, exist_ok=True)
                
                # 用transformers加载模型，它会自动下载
                logging.info(f"加载模型 {model_repo}...")
                config = AutoConfig.from_pretrained(model_repo, cache_dir=target_dir, proxies=proxies)
                model = AutoModel.from_pretrained(model_repo, config=config, cache_dir=target_dir, proxies=proxies)
                tokenizer = AutoTokenizer.from_pretrained(model_repo, cache_dir=target_dir, proxies=proxies)
                
                # 保存到本地
                model.save_pretrained(target_dir)
                tokenizer.save_pretrained(target_dir)
                
                download_path = target_dir
                logging.info(f"模型下载成功并保存到: {target_dir}")
            except Exception as trans_error:
                logging.warning(f"尝试方法2失败: {str(trans_error)}")
                raise Exception(f"多种下载方法均失败: {str(hub_error)}, {str(trans_error)}")
        
        
        logging.info(f"模型下载完成，保存到: {download_path}")
        
        # 验证下载是否成功，检查关键文件是否存在
        essential_files = ["config.json", "pytorch_model.bin"]
        missing_files = [f for f in essential_files if not os.path.exists(os.path.join(target_dir, f))]
        
        if missing_files:
            logging.warning(f"下载完成，但缺少关键文件: {', '.join(missing_files)}")
            return jsonify({
                "success": True,
                "message": f"启动子识别模型 '{model_name}' 下载完成，但可能缺少部分文件",
                "warning": f"模型下载完成，但缺少关键文件: {', '.join(missing_files)}"
            })
        
        return jsonify({
            "success": True,
            "message": f"启动子识别模型 '{model_name}' 下载成功"
        })
        
    except Exception as e:
        logging.error(f"下载启动子模型时出错: {str(e)}")
        
        # 尝试不使用代理直接下载
        try:
            logging.info("代理下载失败，尝试使用直接连接方式...")
            
            # 重置环境变量
            os.environ.pop('http_proxy', None)
            os.environ.pop('https_proxy', None)
            os.environ.pop('HTTP_PROXY', None)
            os.environ.pop('HTTPS_PROXY', None)
            
            # 直接下载，不使用代理
            download_path = snapshot_download(
                repo_id=model_repo,
                local_dir=target_dir,
                local_dir_use_symlinks=False,
                revision="main",
                proxies=None,  # 不使用代理
                resume_download=True
            )
            
            logging.info(f"直接连接下载成功，保存到: {download_path}")
            
            # 验证下载是否成功，检查关键文件是否存在
            essential_files = ["config.json", "pytorch_model.bin"]
            missing_files = [f for f in essential_files if not os.path.exists(os.path.join(target_dir, f))]
            
            if missing_files:
                logging.warning(f"下载完成，但缺少关键文件: {', '.join(missing_files)}")
                return jsonify({
                    "success": True,
                    "message": f"启动子识别模型 '{model_name}' 下载完成 (直接连接方式)，但可能缺少部分文件",
                    "warning": f"模型下载完成，但缺少关键文件: {', '.join(missing_files)}"
                })
            
            return jsonify({
                "success": True,
                "message": f"启动子识别模型 '{model_name}' 下载成功 (直接连接方式)"
            })
            
        except Exception as direct_error:
            logging.error(f"尝试直接连接也失败: {str(direct_error)}")
            
            # 尝试使用requests库直接下载核心文件
            try:
                logging.info("尝试方法3：使用requests直接下载关键文件...")
                import requests
                
                # 确保目录存在
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                
                # 禁用SSL验证
                requests.packages.urllib3.disable_warnings()
                
                # 下载关键文件列表
                files_to_download = [
                    "config.json",
                    "pytorch_model.bin",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "vocab.txt"
                ]
                
                download_success = []
                download_failed = []
                
                # 循环下载所有文件
                for filename in files_to_download:
                    file_url = f"https://huggingface.co/{model_repo}/resolve/main/{filename}"
                    save_path = os.path.join(target_dir, filename)
                    
                    if download_file(file_url, save_path, proxies=None):
                        download_success.append(filename)
                    else:
                        download_failed.append(filename)
                
                # 检查必要文件是否下载成功
                essential_files = ["config.json", "pytorch_model.bin"]
                essential_success = all(f in download_success for f in essential_files)
                
                if essential_success:
                    logging.info(f"核心文件下载成功: {', '.join(download_success)}")
                    if download_failed:
                        return jsonify({
                            "success": True,
                            "message": f"启动子识别模型 '{model_name}' 主要文件下载成功",
                            "warning": f"部分非关键文件下载失败: {', '.join(download_failed)}"
                        })
                    else:
                        return jsonify({
                            "success": True,
                            "message": f"启动子识别模型 '{model_name}' 完全下载成功"
                        })
                elif "config.json" in download_success:
                    logging.warning(f"仅下载成功配置文件，模型文件下载失败")
                    return jsonify({
                        "success": True,
                        "message": f"启动子识别模型 '{model_name}' 部分下载成功 (仅配置文件)",
                        "warning": "仅下载了配置文件，模型文件太大，请检查网络连接后重试"
                    })
                else:
                    logging.error(f"关键文件下载失败")
            
            except Exception as requests_error:
                logging.error(f"使用requests下载也失败: {str(requests_error)}")
            
            # 全部尝试失败，清理目录
            if os.path.exists(target_dir):
                try:
                    shutil.rmtree(target_dir)
                except Exception as cleanup_error:
                    logging.error(f"清理目录失败: {str(cleanup_error)}")
            
            return jsonify({
                "success": False,
                "error": f"下载模型失败: 代理连接错误，请检查Clash代理设置（端口7890）或尝试关闭代理后重试"
            }), 500


@app.route('/models/pull/embedding', methods=['POST'])
def pull_embedding_model():
    """从Hugging Face拉取文本向量化模型"""
    import os
    from huggingface_hub import snapshot_download
    try:
        # 配置模型下载相关参数
        model_repo = "shibing624/text2vec-base-chinese"
        model_dir = "embedding_model"
        model_name = "shibing624_text2vec-base-chinese"  # 模型名称
        target_dir = os.path.join(model_dir, model_name)
        # 检查目录是否已存在
        if os.path.exists(target_dir):
            return jsonify({
                "success": False,
                "message": f"向量化模型 '{model_name}' 已存在",
                "error": "模型已存在，如需更新请先删除现有模型"
            }), 400
        
        # 确保目录结构存在
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        logging.info(f"开始从 Hugging Face 下载向量化模型: {model_repo}")
        
        # 配置代理和SSL设置
        os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'
        os.environ['CURL_CA_BUNDLE'] = ''  # 禁用证书验证
        
        # 设置系统环境变量代理
        os.environ['http_proxy'] = 'http://127.0.0.1:7890'
        os.environ['https_proxy'] = 'http://127.0.0.1:7890'
        os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
        os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
        
        # 配置使用 Clash 代理 (7890端口)
        proxies = {
            "http": "http://127.0.0.1:7890",
            "https": "http://127.0.0.1:7890"
        }
        
        logging.info(f"正在使用本地代理 (端口7890) 下载模型")
        
        # 使用 huggingface_hub 的 snapshot_download 函数一次性下载整个模型仓库
        try:
            logging.info("尝试方法1：使用huggingface_hub的snapshot_download")
            download_path = snapshot_download(
                repo_id=model_repo,
                local_dir=target_dir,
                local_dir_use_symlinks=False,  # 不使用符号链接，直接下载文件
                revision="main",  # 使用主分支
                proxies=proxies,  # 使用设置的代理
                resume_download=True,  # 启用断点续传
                force_download=False,  # 避免强制重新下载
                max_workers=1  # 减少并行下载数量，提高稳定性
            )
        except Exception as hub_error:
            logging.warning(f"尝试方法1失败: {str(hub_error)}")
            
            # 方法2：使用transformers的自动下载功能
            try:
                logging.info("尝试方法2：使用transformers的AutoModel下载")
                from transformers import AutoModel, AutoTokenizer
                
                # 创建目录
                os.makedirs(target_dir, exist_ok=True)
                
                # 用transformers加载模型，它会自动下载
                logging.info(f"加载模型 {model_repo}...")
                model = AutoModel.from_pretrained(model_repo, cache_dir=target_dir, proxies=proxies)
                tokenizer = AutoTokenizer.from_pretrained(model_repo, cache_dir=target_dir, proxies=proxies)
                
                # 保存到本地
                model.save_pretrained(target_dir)
                tokenizer.save_pretrained(target_dir)
                
                download_path = target_dir
                logging.info(f"模型下载成功并保存到: {target_dir}")
            except Exception as trans_error:
                logging.warning(f"尝试方法2失败: {str(trans_error)}")
                raise Exception(f"多种下载方法均失败: {str(hub_error)}, {str(trans_error)}")
        
        
        logging.info(f"模型下载完成，保存到: {download_path}")
        
        # 验证下载是否成功，检查关键文件是否存在
        essential_files = ["config.json", "pytorch_model.bin", "tokenizer_config.json"]
        missing_files = [f for f in essential_files if not os.path.exists(os.path.join(target_dir, f))]
        
        if missing_files:
            logging.warning(f"下载完成，但缺少关键文件: {', '.join(missing_files)}")
            return jsonify({
                "success": True,
                "message": f"向量化模型 '{model_name}' 下载完成，但可能缺少部分文件",
                "warning": f"模型下载完成，但缺少关键文件: {', '.join(missing_files)}"
            })
        
        return jsonify({
            "success": True,
            "message": f"向量化模型 '{model_name}' 下载成功"
        })
        
    except Exception as e:
        logging.error(f"下载向量化模型时出错: {str(e)}")
        
        # 尝试不使用代理直接下载
        try:
            logging.info("代理下载失败，尝试使用直接连接方式...")
            
            # 重置环境变量
            os.environ.pop('http_proxy', None)
            os.environ.pop('https_proxy', None)
            os.environ.pop('HTTP_PROXY', None)
            os.environ.pop('HTTPS_PROXY', None)
            
            # 直接下载，不使用代理
            download_path = snapshot_download(
                repo_id=model_repo,
                local_dir=target_dir,
                local_dir_use_symlinks=False,
                revision="main",
                proxies=None,  # 不使用代理
                resume_download=True
            )
            
            logging.info(f"直接连接下载成功，保存到: {download_path}")
            
            # 验证下载是否成功，检查关键文件是否存在
            essential_files = ["config.json", "pytorch_model.bin", "tokenizer_config.json"]
            missing_files = [f for f in essential_files if not os.path.exists(os.path.join(target_dir, f))]
            
            if missing_files:
                logging.warning(f"下载完成，但缺少关键文件: {', '.join(missing_files)}")
                return jsonify({
                    "success": True,
                    "message": f"向量化模型 '{model_name}' 下载完成 (直接连接方式)，但可能缺少部分文件",
                    "warning": f"模型下载完成，但缺少关键文件: {', '.join(missing_files)}"
                })
            
            return jsonify({
                "success": True,
                "message": f"向量化模型 '{model_name}' 下载成功 (直接连接方式)"
            })
            
        except Exception as direct_error:
            logging.error(f"尝试直接连接也失败: {str(direct_error)}")
            
            # 尝试使用requests库直接下载核心文件
            try:
                logging.info("尝试方法3：使用requests直接下载关键文件...")
                import requests
                
                # 确保目录存在
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                
                # 禁用SSL验证
                requests.packages.urllib3.disable_warnings()
                
                # 下载关键文件列表
                files_to_download = [
                    "config.json",
                    "pytorch_model.bin",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "vocab.txt"
                ]
                
                download_success = []
                download_failed = []
                
                # 循环下载所有文件
                for filename in files_to_download:
                    file_url = f"https://huggingface.co/{model_repo}/resolve/main/{filename}"
                    save_path = os.path.join(target_dir, filename)
                    
                    if download_file(file_url, save_path, proxies=None):
                        download_success.append(filename)
                    else:
                        download_failed.append(filename)
                
                # 检查必要文件是否下载成功
                essential_files = ["config.json", "pytorch_model.bin", "tokenizer_config.json"]
                essential_success = all(f in download_success for f in essential_files)
                
                if essential_success:
                    logging.info(f"核心文件下载成功: {', '.join(download_success)}")
                    if download_failed:
                        return jsonify({
                            "success": True,
                            "message": f"向量化模型 '{model_name}' 主要文件下载成功",
                            "warning": f"部分非关键文件下载失败: {', '.join(download_failed)}"
                        })
                    else:
                        return jsonify({
                            "success": True,
                            "message": f"向量化模型 '{model_name}' 完全下载成功"
                        })
                elif len(download_success) > 0:
                    logging.warning(f"部分文件下载成功：{', '.join(download_success)}，但缺少核心文件")
                    return jsonify({
                        "success": True,
                        "message": f"向量化模型 '{model_name}' 部分下载成功 (仅部分配置文件)",
                        "warning": "未能下载所有必要文件，请检查网络连接后重试"
                    })
                else:
                    logging.error(f"所有关键文件下载失败")
            
            except Exception as requests_error:
                logging.error(f"使用requests下载也失败: {str(requests_error)}")
            
            # 全部尝试失败，清理目录
            if os.path.exists(target_dir):
                try:
                    shutil.rmtree(target_dir)
                except Exception as cleanup_error:
                    logging.error(f"清理目录失败: {str(cleanup_error)}")
            
            return jsonify({
                "success": False,
                "error": f"下载模型失败: 代理连接错误，请检查Clash代理设置（端口7890）或尝试关闭代理后重试"
            }), 500
def download_file(url, save_path, proxies=None):
    """直接下载文件，支持代理设置和进度反馈"""
    try:
        logging.info(f"直接下载文件: {url} -> {save_path}")
        
        # 创建保存路径的目录（如果不存在）
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 发送GET请求下载文件
        response = requests.get(
            url, 
            stream=True,  # 流式下载大文件
            proxies=proxies,
            verify=False,  # 禁用SSL验证
            timeout=60
        )
        response.raise_for_status()  # 如果响应码不是200，抛出异常
        
        # 获取文件总大小
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        # 写入文件
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    # 每下载10%记录一次日志
                    if total_size > 0 and downloaded_size % (total_size // 10) < 8192:
                        percent = (downloaded_size / total_size) * 100
                        logging.info(f"下载进度: {percent:.1f}% ({downloaded_size}/{total_size})")
        
        logging.info(f"文件下载完成: {save_path}")
        return True
    except Exception as e:
        logging.error(f"文件下载失败: {str(e)}")
        return False

# ============================
# 报告管理模块方法
# ============================
@app.route('/getAllSummaries', methods=['GET'])
def get_all_summaries():
    try:
        files = os.listdir(SUMMARY_FOLDER)
        return jsonify(files)
    except Exception as e:
        return jsonify({"error": f"Failed to get summaries: {str(e)}"}), 500
@app.route('/reports/<report_filename>', methods=['DELETE'])
def delete_report(report_filename):
    """删除指定报告文件"""
    try:
        # 验证文件名格式
        if not report_filename.endswith('.txt'):
            return jsonify({"error": "Invalid report filename"}), 400
            
        # 构建文件路径
        report_path = os.path.join(SUMMARY_FOLDER, report_filename)
        
        # 检查文件是否存在
        if not os.path.exists(report_path):
            return jsonify({"error": "Report file not found"}), 404
            
        # 删除报告文件
        os.remove(report_path)
        logging.info(f"Deleted report file: {report_path}")
        
        # 同时删除对应的GFF文件（如果存在）
        # 从报告文件名中提取genome_id和日期
        filename_parts = report_filename.replace('_基因组分析报告_', '_').replace('.txt', '').split('_')
        if len(filename_parts) >= 2:
            genome_id = filename_parts[0]
            date = filename_parts[1]
            gff_filename = f"{genome_id}_annotation_{date}.gff"
            gff_path = os.path.join(GFF_FOLDER, gff_filename)
            
            if os.path.exists(gff_path):
                os.remove(gff_path)
                logging.info(f"Deleted corresponding GFF file: {gff_path}")
        
        return jsonify({"message": "Report deleted successfully"})
        
    except Exception as e:
        logging.error(f"Failed to delete report {report_filename}: {str(e)}")
        return jsonify({"error": f"Failed to delete report: {str(e)}"}), 500

@app.route('/reports/<report_filename>/details', methods=['GET'])
def get_report_details(report_filename):
    """获取报告详细信息"""
    try:
        # 验证文件名格式
        if not report_filename.endswith('.txt'):
            return jsonify({"error": "Invalid report filename"}), 400
            
        report_path = os.path.join(SUMMARY_FOLDER, report_filename)
        
        # 检查文件是否存在
        if not os.path.exists(report_path):
            return jsonify({"error": "Report file not found"}), 404
            
        # 获取文件基本信息
        file_stats = os.stat(report_path)
        file_size = file_stats.st_size
        file_modified = file_stats.st_mtime
        
        # 格式化文件大小
        if file_size < 1024:
            size_str = f"{file_size} B"
        elif file_size < 1024*1024:
            size_str = f"{file_size/1024:.1f} KB"
        else:
            size_str = f"{file_size/(1024*1024):.1f} MB"
            
        # 从文件名解析信息
        filename_parts = report_filename.replace('_基因组分析报告_', '_').replace('.txt', '').split('_')
        genome_id = filename_parts[0] if len(filename_parts) > 0 else '未知'
        date = filename_parts[1] if len(filename_parts) > 1 else '未知'
        
        # 检查是否有对应的GFF文件
        gff_filename = f"{genome_id}_annotation_{date}.gff"
        gff_path = os.path.join(GFF_FOLDER, gff_filename)
        has_gff = os.path.exists(gff_path)
        
        report_details = {
            "filename": report_filename,
            "genome_id": genome_id,
            "analysis_date": date,
            "file_size": size_str,
            "file_size_bytes": file_size,
            "last_modified": file_modified,
            "has_gff": has_gff,
            "gff_filename": gff_filename if has_gff else None
        }
        
        return jsonify(report_details)
        
    except Exception as e:
        logging.error(f"Failed to get report details for {report_filename}: {str(e)}")
        return jsonify({"error": f"Failed to get report details: {str(e)}"}), 500
    
# ============================
# 模型训练模块方法
# ============================
@app.route('/save_model', methods=['POST'])
def save_model():
    """保存训练好的模型和配置到model目录，不覆盖同名文件"""
    try:
        model_file = request.files.get('model_file')
        config_json = request.form.get('config_json')
        model_name = request.form.get('model_name')
        if not model_file or not config_json or not model_name:
            return jsonify({"error": "缺少参数"}), 400
        # 只允许字母数字下划线
        import re
        if not re.match(r'^\w+$', model_name):
            return jsonify({"error": "模型名只能包含字母、数字和下划线"}), 400
        pth_path = os.path.join('model', model_name + '.pth')
        json_path = os.path.join('model', model_name + '.json')
        if os.path.exists(pth_path) or os.path.exists(json_path):
            return jsonify({"error": "同名模型已存在"}), 409
        # 保存pth
        model_file.save(pth_path)
        # 保存json
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(config_json)
        return jsonify({"message": "模型保存成功", "pth": pth_path, "json": json_path})
    except Exception as e:
        logging.error(f"Failed to save model: {str(e)}")
        return jsonify({"error": f"Failed to save model: {str(e)}"}), 500

@app.route('/train_model', methods=['POST'])
def train_model():
    """模型训练接口，流式返回每个epoch评估信息，训练完成后返回模型和json"""
    import tempfile, shutil, json, logging, os
    from train import ProteinClassifier
    def stream_train():
        try:
            logging.info('收到/train_model请求')
            pretrain_model = request.form.get('pretrain_model', None)
            new_model_name = request.form.get('new_model_name', None)
            model_type = request.form.get('model_type', 'HybridModel')
            epochs = int(request.form.get('epochs', 10))
            batch_size = int(request.form.get('batch_size', 32))
            learning_rate = float(request.form.get('learning_rate', 1e-4))
            max_length = int(request.form.get('max_length', 10000))
            threshold = float(request.form.get('threshold', 0.5))
            val_ratio = float(request.form.get('val_ratio', 0.2))
            files = request.files.getlist('file')
            if not files or len(files) == 0:
                yield json.dumps({"error": "未上传任何数据文件"}) + '\n'
                return
            temp_dir = tempfile.mkdtemp()
            for f in files:
                save_path = os.path.join(temp_dir, os.path.basename(f.filename))
                f.save(save_path)
            config = {
                'max_length': max_length,
                'batch_size': batch_size,
                'epochs': epochs,
                'threshold': threshold,
                'learning_rate': learning_rate
            }
            try:
                if pretrain_model:
                    pretrain_pth = os.path.join('model', pretrain_model + '.pth') if not pretrain_model.endswith('.pth') else os.path.join('model', pretrain_model)
                    pretrain_json = pretrain_pth.replace('.pth', '.json')
                    classifier = ProteinClassifier.per_train_load(model_path=pretrain_pth, config=config, model_info_path=pretrain_json, model=model_type)
                else:
                    classifier = ProteinClassifier(config=config, train_model=model_type)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_pth:
                    model_path = tmp_pth.name
                with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_json:
                    config_path = tmp_json.name
                # 流式训练
                def epoch_callback(epoch_info):
                    yield_data = json.dumps({"type": "epoch", **epoch_info}) + '\n'
                    yield yield_data
                # 用生成器收集epoch信息
                epoch_logs = []
                def collect_epoch(epoch_info):
                    epoch_logs.append(epoch_info)
                    yield_data = json.dumps({"type": "epoch", **epoch_info}) + '\n'
                    yield yield_data
                # 由于Python yield嵌套，需用队列传递
                import queue
                q = queue.Queue()
                def cb(epoch_info):
                    q.put(json.dumps({"type": "epoch", **epoch_info}) + '\n')
                # 启动训练
                def train_and_yield():
                    classifier.train(temp_dir, save_path=model_path, json_path=config_path, val_ratio=val_ratio, epoch_callback=cb)
                    # 训练完成后输出模型
                    with open(model_path, 'rb') as f:
                        model_bytes = f.read()
                    with open(config_path, 'r', encoding='utf-8') as f:
                        model_json = json.load(f)
                    model_hex = model_bytes.hex()
                    if new_model_name:
                        import re
                        if not re.match(r'^\w+$', new_model_name):
                            q.put(json.dumps({"error": "模型名只能包含字母、数字和下划线"}) + '\n')
                            return
                        pth_path = os.path.join('model', new_model_name + '.pth')
                        json_path = os.path.join('model', new_model_name + '.json')
                        if os.path.exists(pth_path) or os.path.exists(json_path):
                            q.put(json.dumps({"error": "同名模型已存在"}) + '\n')
                            return
                        shutil.copy(model_path, pth_path)
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(model_json, f, ensure_ascii=False, indent=2)
                    os.remove(model_path)
                    os.remove(config_path)
                    shutil.rmtree(temp_dir)
                    q.put(json.dumps({
                        "type": "done",
                        "model_bytes": model_hex,
                        "model_json": model_json
                    }) + '\n')
                import threading
                t = threading.Thread(target=train_and_yield)
                t.start()
                while True:
                    try:
                        msg = q.get(timeout=0.5)
                        yield msg
                        if '"type": "done"' in msg or '"error"' in msg:
                            break
                    except queue.Empty:
                        if not t.is_alive():
                            break
            except Exception as e:
                shutil.rmtree(temp_dir)
                yield json.dumps({"error": f"训练失败: {str(e)}"}) + '\n'
        except Exception as e:
            yield json.dumps({"error": f"训练接口异常: {str(e)}"}) + '\n'
    return Response(stream_with_context(stream_train()), mimetype='text/plain')

# ============================
# 知识库导入模块方法
# ============================
import sys
sys.path.append('.')
from creat_knowledgeDB import add_to_knowledge_base, list_collections, get_collection_info
# 知识库临时文件清理函数
def cleanup_knowledge_files():
    """清理knowledge_files文件夹中的临时文件"""
    try:
        knowledge_dir = os.path.abspath(KNOWLEDGE_FOLDER)
        if os.path.exists(knowledge_dir):
            # 删除文件夹中的所有文件
            files_deleted = 0
            for filename in os.listdir(knowledge_dir):
                file_path = os.path.join(knowledge_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    files_deleted += 1
                    logging.info(f"清理知识库临时文件: {file_path}")
            
            # 如果文件夹为空，删除文件夹
            if len(os.listdir(knowledge_dir)) == 0:
                os.rmdir(knowledge_dir)
                logging.info(f"删除空的知识库文件夹: {knowledge_dir}")
                return True, files_deleted
            return False, files_deleted
        return False, 0
    except Exception as e:
        logging.warning(f"清理知识库文件夹时出错: {str(e)}")
        return False, 0

# 检查文件扩展名是否允许
def allowed_knowledge_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'txt', 'pdf', 'docx', 'doc', 'csv', 'json'}

# 将知识文件转换为txt格式
def convert_to_txt(file_path, output_path):
    """简单的文件转换，将各种文件转为txt格式"""
    # 确保文件存在
    if not os.path.exists(file_path):
        logging.error(f"转换文件不存在: {file_path}")
        return False
        
    # 确保目标目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext == '.txt':
        # 直接复制txt文件，但需要检查源和目标是否相同
        try:
            if os.path.abspath(file_path) == os.path.abspath(output_path):
                # 源和目标相同，无需复制
                logging.info(f"TXT文件源路径和目标路径相同，无需复制: {file_path}")
                return True
            else:
                shutil.copy2(file_path, output_path)
                logging.info(f"已复制TXT文件: {file_path} -> {output_path}")
                return True
        except Exception as e:
            logging.error(f"复制文件失败: {str(e)}")
            return False
    
    if ext == '.pdf':
        try:
            import PyPDF2
            with open(file_path, 'rb') as pdf_file, open(output_path, 'w', encoding='utf-8') as txt_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    txt_file.write(page.extract_text() + '\n\n')
            return True
        except Exception as e:
            logging.error(f"PDF转换错误: {str(e)}")
            return False
            
    if ext == '.docx':
        try:
            import docx
            doc = docx.Document(file_path)
            with open(output_path, 'w', encoding='utf-8') as txt_file:
                for para in doc.paragraphs:
                    txt_file.write(para.text + '\n')
            return True
        except Exception as e:
            logging.error(f"DOCX转换错误: {str(e)}")
            return False
    
    if ext == '.doc':
        # 对于旧版doc文件，可能需要额外的库
        logging.error("暂不支持.doc格式，请转换为.docx后再上传")
        return False
        
    if ext in ['.json', '.csv']:
        # 简单地复制内容
        shutil.copy2(file_path, output_path)
        return True
        
    return False

@app.route('/api/knowledge', methods=['POST'])
def add_knowledge():
    """上传知识文件并添加到知识库"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if not allowed_knowledge_file(file.filename):
        return jsonify({'error': 'File type not allowed. Supported types: txt, pdf, docx, doc, csv, json'}), 400
    
    try:
        # 确保知识库文件夹存在，使用绝对路径
        knowledge_dir = os.path.abspath(KNOWLEDGE_FOLDER)
        os.makedirs(knowledge_dir, exist_ok=True)
        logging.info(f"知识库文件夹路径: {knowledge_dir}")
        
        # 保存上传的文件
        upload_path = os.path.join(knowledge_dir, file.filename)
        logging.info(f"保存上传文件到: {upload_path}")
        file.save(upload_path)
        
        # 检查文件是否成功保存
        if not os.path.exists(upload_path):
            return jsonify({'error': f'无法保存文件，请检查路径权限: {upload_path}'}), 500
        
        # 判断是否已经是txt格式
        _, file_ext = os.path.splitext(file.filename)
        if file_ext.lower() == '.txt':
            # 如果已经是txt文件，直接使用它
            txt_filename = file.filename
            txt_path = upload_path
            logging.info(f"已上传txt文件，直接使用: {txt_path}")
            conversion_success = True
        else:
            # 需要转换为txt格式
            txt_filename = f"{os.path.splitext(file.filename)[0]}.txt"
            txt_path = os.path.join(knowledge_dir, txt_filename)
            logging.info(f"转换后的文件路径: {txt_path}")
            
            conversion_success = convert_to_txt(upload_path, txt_path)
            if not conversion_success:
                return jsonify({'error': f'Failed to convert {file.filename} to txt format'}), 500
            
            # 如果原文件不是txt格式，删除原文件（可选）
            os.remove(upload_path)
            
        # 将转换后的文本添加到知识库
        try:
            # 确保文件存在
            if not os.path.exists(txt_path):
                logging.error(f"文件未成功保存: {txt_path}")
                return jsonify({'error': f'文件保存失败: {txt_path}'}), 500
                
            logging.info(f"开始添加文件到知识库: {txt_path}")
            try:
                # 确认文件路径为绝对路径
                abs_txt_path = os.path.abspath(txt_path)
                logging.info(f"使用绝对路径添加到知识库: {abs_txt_path}")
                
                # 验证文件可读性
                try:
                    with open(abs_txt_path, 'r', encoding='utf-8') as f:
                        sample = f.read(100)  # 尝试读取部分内容以验证文件可访问
                        logging.info(f"文件内容验证成功，前100字符: {sample[:50]}...")
                except Exception as read_error:
                    logging.error(f"无法读取文件: {str(read_error)}")
                    return jsonify({'error': f'无法读取文件: {str(read_error)}'}), 500
                
                # 添加到知识库
                add_to_knowledge_base(abs_txt_path)
                logging.info(f"成功添加文件到知识库")
                
                # 删除处理后的文件并清理文件夹
                try:
                    if os.path.exists(abs_txt_path):
                        os.remove(abs_txt_path)
                        logging.info(f"成功删除处理后的文件: {abs_txt_path}")
                    
                    # 使用通用清理函数清理其他可能的临时文件
                    folder_deleted, files_deleted = cleanup_knowledge_files()
                    if files_deleted > 0:
                        logging.info(f"额外清理了 {files_deleted} 个临时文件")
                except Exception as del_err:
                    logging.warning(f"删除文件时出错，但不影响处理结果: {str(del_err)}")
            except Exception as e:
                logging.error(f"添加知识库时出错: {str(e)}")
                return jsonify({'error': f'知识库处理失败: {str(e)}'}), 500
            
            # 使用清理函数做最后的检查
            folder_deleted, _ = cleanup_knowledge_files()
            if folder_deleted:
                logging.info(f"知识库临时文件夹已完全清理")
            else:
                logging.info(f"知识库临时文件夹可能仍有其他文件存在")
            
            # 返回成功响应
            return jsonify({
                'success': True,
                'message': f'知识文件 {file.filename} 已成功添加到知识库',
                'filename': txt_filename
            })
            
        except Exception as e:
            logging.error(f"添加知识库失败: {str(e)}")
            return jsonify({'error': f'知识库处理失败: {str(e)}'}), 500
            
    except Exception as e:
        logging.error(f"文件处理错误: {str(e)}")
        
        # 出错时也尝试清理临时文件
        try:
            if 'upload_path' in locals() and os.path.exists(upload_path):
                os.remove(upload_path)
                logging.info(f"已清理上传的临时文件: {upload_path}")
                
            if 'txt_path' in locals() and os.path.exists(txt_path) and txt_path != upload_path:
                os.remove(txt_path)
                logging.info(f"已清理转换后的临时文件: {txt_path}")
                
            # 如果文件夹为空，删除文件夹
            if 'knowledge_dir' in locals() and os.path.exists(knowledge_dir) and len(os.listdir(knowledge_dir)) == 0:
                os.rmdir(knowledge_dir)
                logging.info(f"已删除空的知识库文件夹: {knowledge_dir}")
        except Exception as cleanup_err:
            logging.warning(f"清理临时文件时出错: {str(cleanup_err)}")
            
        return jsonify({'error': f'处理文件时出错: {str(e)}'}), 500

@app.route('/api/knowledge/collections', methods=['GET'])
def get_knowledge_collections():
    """获取知识库中的集合信息"""
    try:
        collections = list_collections()
        
        # 在查询集合信息的同时，确保knowledge_files文件夹已清理
        folder_deleted, files_deleted = cleanup_knowledge_files()
        if files_deleted > 0:
            logging.info(f"已清理 {files_deleted} 个临时文件")
        if folder_deleted:
            logging.info(f"已删除临时文件夹 {KNOWLEDGE_FOLDER}")
        
        return jsonify({
            'success': True,
            'collections': [{'name': c.name, 'id': c.id} for c in collections]
        })
    except Exception as e:
        logging.error(f"获取知识库集合失败: {str(e)}")
        return jsonify({'error': f'获取知识库集合失败: {str(e)}'}), 500

# 启动Flask应用
if __name__ == '__main__':
    print("启动Flask服务器...")
    app.run(host='0.0.0.0', port=5000, debug=True)
    print("服务器已关闭")


