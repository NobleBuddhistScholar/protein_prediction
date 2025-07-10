from flask import Flask
from flask_cors import CORS
from config import init_folders, UPLOAD_FOLDER, RESULT_FOLDER, \
    SUMMARY_FOLDER, KNOWLEDGE_FOLDER, GFF_FOLDER, ALLOWED_EXTENSIONS

# 初始化文件夹（会在instance下创建）
#init_folders()

app = Flask(__name__)
CORS(app)

# 注入配置到app
app.config.update({
    "UPLOAD_FOLDER": UPLOAD_FOLDER,
    "RESULT_FOLDER": RESULT_FOLDER,
    "SUMMARY_FOLDER": SUMMARY_FOLDER,
    "KNOWLEDGE_FOLDER": KNOWLEDGE_FOLDER,
    "GFF_FOLDER": GFF_FOLDER,
    "ALLOWED_EXTENSIONS": ALLOWED_EXTENSIONS
})

# 导入所有蓝图
from routes.gene_visualization import gene_bp
from routes.knowledge_manage import knowledge_bp
from routes.model_manage import model_bp
from routes.model_train import train_bp
from routes.summary_report import report_bp
from routes.commence import commence_bp

# 注册蓝图（URL前缀可自定义）
app.register_blueprint(gene_bp, url_prefix='/gene')
app.register_blueprint(knowledge_bp, url_prefix='/knowledge')
app.register_blueprint(model_bp, url_prefix='/model')
app.register_blueprint(train_bp, url_prefix='/train')
app.register_blueprint(report_bp, url_prefix='/report')
app.register_blueprint(commence_bp, url_prefix='/commence')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)