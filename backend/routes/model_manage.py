import json
import os
import logging
import requests
from flask import request, jsonify
import shutil
from flask import Blueprint, jsonify

# 创建蓝图对象
model_bp = Blueprint('model', __name__)
# ============================
# 模型管理模块方法
# ============================
@model_bp.route('/models/<model_filename>', methods=['DELETE'])
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

@model_bp.route('/models/<model_filename>/rename', methods=['PUT'])
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

@model_bp.route('/models/<model_filename>/details', methods=['GET'])
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
@model_bp.route('/models/<model_filename>/export', methods=['GET'])
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
        return send_file(model_path, mimetype='model_bplication/octet-stream')
    except Exception as e:
        logging.error(f"Failed to export model {model_filename}: {str(e)}")
        return jsonify({"error": f"Failed to export model: {str(e)}"}), 500

@model_bp.route('/models/pull/promoter', methods=['POST'])
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


@model_bp.route('/models/pull/embedding', methods=['POST'])
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