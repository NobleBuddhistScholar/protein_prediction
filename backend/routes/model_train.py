import os
import logging
from flask import Response, stream_with_context
from flask import request, jsonify
from flask import Blueprint, jsonify

# 创建蓝图对象
train_bp = Blueprint('train', __name__)

# ============================
# 模型训练模块方法
# ============================
@train_bp.route('/save_model', methods=['POST'])
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

@train_bp.route('/train_model', methods=['POST'])
def train_model():
    """模型训练接口，流式返回每个epoch评估信息，训练完成后返回模型和json"""
    import tempfile, shutil, json, logging, os
    from train import ProteinClassifier
    def stream_train():
        try:
            logging.info('收到/train_model请求')
            print(request.form)
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
                    # 处理文件名后缀（确保以.pth结尾）
                    if not pretrain_model.endswith('.pth'):
                        pretrain_model_with_ext = f"{pretrain_model}.pth"
                    else:
                        pretrain_model_with_ext = pretrain_model

                    # 用"/"直接拼接路径，确保分隔符为"/"
                    pretrain_pth = f"model/{pretrain_model_with_ext}"

                    print(pretrain_pth)
                    pretrain_json = pretrain_pth.replace('.pth', '.json')
                    print(pretrain_pth, config, pretrain_json, model_type)
                    import pdb; pdb.set_trace()
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
