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
gene_bp = Blueprint('gene', __name__)

# 配置日志记录
logging.basicConfig(level=logging.DEBUG)

# 配置项
UPLOAD_FOLDER = "prectice_test"
RESULT_FOLDER = "./instance/result"
SUMMARY_FOLDER = "./instance/summary"
ALLOWED_EXTENSIONS = {'fasta'}
# 在配置项部分新增GFF文件夹配置
GFF_FOLDER = "./instance/gff"

# ============================
# 基因组注释模块方法
# ============================
# 检查文件类型
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#上传文件并处理基因组预测
@gene_bp.route('/upload', methods=['POST'])
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
@gene_bp.route('/stream_summary', methods=['POST'])
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

# 利用generate_gff.py生成gff报告
@gene_bp.route('/generate_gff', methods=['GET'])
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