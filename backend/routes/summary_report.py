import os
import logging
from flask import Response, stream_with_context
from flask import request, jsonify
from flask import Blueprint, jsonify

# 创建蓝图对象
report_bp = Blueprint('report', __name__)

# 配置日志记录
logging.basicConfig(level=logging.DEBUG)

# 配置项
SUMMARY_FOLDER = "./instance/summary"
# 在配置项部分新增GFF文件夹配置
GFF_FOLDER = "./instance/gff"

# ============================
# 报告管理模块方法
# ============================
@report_bp.route('/getAllSummaries', methods=['GET'])
def get_all_summaries():
    try:
        files = os.listdir(SUMMARY_FOLDER)
        return jsonify(files)
    except Exception as e:
        return jsonify({"error": f"Failed to get summaries: {str(e)}"}), 500
@report_bp.route('/reports/<report_filename>', methods=['DELETE'])
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

@report_bp.route('/reports/<report_filename>/details', methods=['GET'])
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
@report_bp.route('/save_model', methods=['POST'])
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

@report_bp.route('/train_model', methods=['POST'])
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