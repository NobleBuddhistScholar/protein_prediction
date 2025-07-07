import json
import os
import logging
from openai import OpenAI
from datetime import datetime
from flask import Response, stream_with_context
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from train import ProteinClassifier
from generate_gff import generate_gff
import queue
import threading
import tempfile
import shutil
import base64
from chat_api import generate_summary_stream

# 配置日志记录
logging.basicConfig(level=logging.DEBUG)

# 配置项
UPLOAD_FOLDER = "prectice_test"
#MODEL_PATH = "model/HyperFusionCortex_v1.pth"
RESULT_FOLDER = "result"
SUMMARY_FOLDER = "summary"
# 新增报告文件夹配置
REPORTS_FOLDER = "summary" 
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
    return Response(
        stream_with_context(
            generate_summary_stream(genome_id, genome_data, save_path)
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

@app.route('/getAllSummaries', methods=['GET'])
def get_all_summaries():
    try:
        files = os.listdir(SUMMARY_FOLDER)
        return jsonify(files)
    except Exception as e:
        return jsonify({"error": f"Failed to get summaries: {str(e)}"}), 500

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

# 模型管理相关API接口

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

# 报告管理相关API接口
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


if __name__ == '__main__':
    app.run(debug=True)