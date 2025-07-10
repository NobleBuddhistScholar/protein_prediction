from flask import Blueprint, jsonify
import logging
import json
import os
import logging
from flask import Response, stream_with_context
from flask import request, jsonify
from train import ProteinClassifier
from generate_gff import generate_gff
from chat_api import generate_summary_stream

# 创建蓝图对象
commence_bp = Blueprint('commence', __name__)


GFF_FOLDER = "./instance/gff"
SUMMARY_FOLDER = "./instance/summary"
# ============================
# 通用方法
# ============================
# 新增：获取模型列表的接口
@commence_bp.route('/models', methods=['GET'])
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
    
# 获取报告
@commence_bp.route('/summary', methods=['GET'])
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
    
# 获取/gff路径下的gff报告
@commence_bp.route('/gff', methods=['GET'])
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