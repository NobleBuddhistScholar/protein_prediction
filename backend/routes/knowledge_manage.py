import os
import logging
from flask import request, jsonify
import shutil
from flask import Blueprint, jsonify

# 创建蓝图对象
knowledge_bp = Blueprint('knowledge', __name__)

# 配置日志记录
logging.basicConfig(level=logging.DEBUG)

# 知识库文件夹配置
KNOWLEDGE_FOLDER = "./instance/knowledge_files" 

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

@knowledge_bp.route('/knowledge', methods=['POST'])
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

@knowledge_bp.route('/knowledge/collections', methods=['GET'])
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