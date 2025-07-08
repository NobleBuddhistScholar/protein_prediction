<template>
  <div class="container">
    <div class="header-section">
      <div>
        <h2 class="section-title">
          <i class="fa-solid fa-brain"></i> 知识库管理
        </h2>
      </div>
      <button class="refresh-btn" @click="refreshPage" :disabled="isUploading">
        <i class="fa-solid fa-sync" :class="{ 'fa-spin': isUploading }"></i> 刷新
      </button>
    </div>

    <!-- 上传区域 -->
    <div class="upload-section">
      <div class="file-upload-container">
        <label class="file-upload-button" :class="{ 'disabled': isUploading }">
          <input 
            type="file" 
            ref="fileInput" 
            @change="handleFileUpload" 
            accept=".txt,.pdf,.docx,.doc,.csv,.json"
            :disabled="isUploading"
          />
          <span v-if="!isUploading && !uploadError">
            <i class="fa-solid fa-cloud-upload-alt"></i> {{ fileName || '选择知识文件' }}
          </span>
          <span v-if="isUploading" class="uploading-text">
            <i class="fa-solid fa-spinner fa-spin"></i> 正在处理中...
          </span>
          <span v-if="uploadError" class="error-text">
            <i class="fa-solid fa-exclamation-triangle"></i> {{ uploadError }}
          </span>
        </label>
        <div v-if="isUploading" class="progress-container">
          <div class="progress-bar" :style="{ width: progress + '%' }"></div>
          <span class="progress-text">{{ progress }}%</span>
        </div>
      </div>

      <div class="upload-actions">
        <button 
          class="action-btn upload-btn" 
          @click="uploadFile" 
          :disabled="!fileToUpload || isUploading"
        >
          <i class="fa-solid fa-upload"></i> 添加到知识库
        </button>
        <button class="action-btn clear-btn" @click="clearSelection" :disabled="isUploading">
          <i class="fa-solid fa-times"></i> 清除
        </button>
      </div>
    </div>

    <!-- 加载状态 -->
    <div v-if="isUploading && !uploadSuccess" class="loading-section">
      <i class="fa-solid fa-spinner fa-spin"></i> 处理中...
    </div>

    <!-- 成功消息 -->
    <div v-if="uploadSuccess" class="success-message">
      <i class="fa-solid fa-check-circle"></i> 
      文件 <strong>{{ uploadedFileName }}</strong> 已成功添加到知识库！
    </div>


    <!-- 文件格式提示 -->
    <div class="format-info">
      <h3 class="sub-section-title">支持的文件格式</h3>
      <div class="format-list">
        <div class="format-item">
          <i class="fa-solid fa-file-alt"></i> TXT文本文件
        </div>
        <div class="format-item">
          <i class="fa-solid fa-file-pdf"></i> PDF文档
        </div>
        <div class="format-item">
          <i class="fa-solid fa-file-word"></i> Word文档
        </div>
        <div class="format-item">
          <i class="fa-solid fa-file-csv"></i> CSV数据文件
        </div>
        <div class="format-item">
          <i class="fa-solid fa-file-code"></i> JSON数据文件
        </div>
      </div>
    </div>

    <!-- 通知消息 -->
    <div v-if="notification.show" :class="['notification', notification.type]">
      <i :class="notificationIcon"></i>
      {{ notification.message }}
    </div>
  </div>
</template>

<script>
import axios from 'axios';
import { API_BASE_URL } from '../config.js';

export default {
  name: 'KnowledgeManage',
  data() {
    return {
      fileToUpload: null,
      fileName: '',
      isUploading: false,
      uploadError: null,
      uploadSuccess: false,
      uploadedFileName: '',
      progress: 0,
      collections: [],
      
      // 通知系统
      notification: {
        show: false,
        message: '',
        type: 'success' // success, error, warning
      },
      
      // 通知定时器
      notificationTimer: null
    };
  },
  
  computed: {
    notificationIcon() {
      const icons = {
        success: 'fa-solid fa-check-circle',
        error: 'fa-solid fa-exclamation-circle',
        warning: 'fa-solid fa-exclamation-triangle'
      };
      return icons[this.notification.type] || icons.success;
    }
  },
  
  mounted() {
    this.getCollections();
  },
  
  methods: {
    handleFileUpload(event) {
      const files = event.target.files;
      if (files.length > 0) {
        this.fileToUpload = files[0];
        this.fileName = files[0].name;
        this.uploadError = null;
        this.uploadSuccess = false;
      }
    },
    
    clearSelection() {
      this.fileToUpload = null;
      this.fileName = '';
      this.uploadError = null;
      this.uploadSuccess = false;
      this.progress = 0;
      if (this.$refs.fileInput) {
        this.$refs.fileInput.value = '';
      }
    },
    
    uploadFile() {
      if (!this.fileToUpload) {
        this.showNotification('请先选择文件', 'warning');
        return;
      }

      const allowedTypes = [
        'text/plain',
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/msword',
        'text/csv',
        'application/json'
      ];

      if (!allowedTypes.includes(this.fileToUpload.type)) {
        this.showNotification('不支持的文件类型，请选择TXT、PDF、DOCX、DOC、CSV或JSON文件', 'error');
        return;
      }

      this.isUploading = true;
      this.uploadError = null;
      this.uploadSuccess = false;
      this.progress = 0;

      const formData = new FormData();
      formData.append('file', this.fileToUpload);

      // 创建进度更新的模拟器（因为后端处理时间可能较长）
      const progressInterval = setInterval(() => {
        if (this.progress < 90) {
          this.progress += Math.floor(Math.random() * 5) + 1;
        }
      }, 300);

      axios.post(`${API_BASE_URL}/api/knowledge`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })
      .then(response => {
        clearInterval(progressInterval);
        this.progress = 100;
        this.isUploading = false;
        this.uploadSuccess = true;
        this.uploadedFileName = this.fileName;
        this.clearSelection();
        this.getCollections(); // 刷新集合列表
        this.showNotification(`文件 ${this.uploadedFileName} 已成功添加到知识库！`, 'success');
      })
      .catch(error => {
        clearInterval(progressInterval);
        this.isUploading = false;
        this.progress = 0;
        
        console.error('文件上传错误:', error);
        
        let errorMessage = '添加知识文件失败，请重试';
        
        // 详细的错误处理
        if (error.response) {
          // 服务器返回了错误响应
          console.error('服务器响应错误:', error.response);
          if (error.response.data && error.response.data.error) {
            errorMessage = error.response.data.error;
          } else {
            errorMessage = `服务器错误 (${error.response.status}): ${error.response.statusText}`;
          }
        } else if (error.request) {
          // 请求已发送但没有收到响应
          console.error('请求未收到响应:', error.request);
          errorMessage = '服务器没有响应，请检查网络连接或联系管理员';
        } else {
          // 请求设置时出了问题
          console.error('请求配置错误:', error.message);
          errorMessage = `请求错误: ${error.message}`;
        }
        
        this.uploadError = errorMessage;
        this.showNotification(errorMessage, 'error');
      });
    },
    
    getCollections() {
      this.isUploading = true;
      
      axios.get(`${API_BASE_URL}/api/knowledge/collections`)
        .then(response => {
          if (response.data.success) {
            this.collections = response.data.collections;
          }
          this.isUploading = false;
        })
        .catch(error => {
          console.error('获取知识库集合失败:', error);
          this.showNotification('获取知识库集合失败', 'error');
          this.isUploading = false;
        });
    },
    
    // 刷新整个页面的方法
    refreshPage() {
      this.isUploading = true;
      
      // 显示加载状态
      setTimeout(() => {
        // 重置所有状态数据
        this.resetAll();      
        // 恢复状态
        this.isUploading = false;
      }, 500);
    },
    
    // 重置所有状态数据的方法
    resetAll() {
      // 重置文件相关状态
      this.fileToUpload = null;
      this.fileName = '';
      this.uploadError = null;
      this.uploadSuccess = false;
      this.uploadedFileName = '';
      this.progress = 0;
      
      // 清除文件输入
      if (this.$refs.fileInput) {
        this.$refs.fileInput.value = '';
      }
      
      // 重置通知状态
      if (this.notificationTimer) {
        clearTimeout(this.notificationTimer);
      }
      this.notification.show = false;
    },
    
    showNotification(message, type = 'success', duration = 3000) {
      // 清除之前的计时器
      if (this.notificationTimer) {
        clearTimeout(this.notificationTimer);
      }
      
      // 设置新的通知
      this.notification = {
        show: true,
        message,
        type
      };
      
      // 设置自动关闭
      this.notificationTimer = setTimeout(() => {
        this.notification.show = false;
      }, duration);
    }
  }
};
</script>

<style scoped>
.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2.5rem 1.5rem 2rem 1.5rem;
  font-family: 'Segoe UI', 'Arial', sans-serif;
  background: linear-gradient(120deg, #f5f7fa 0%, #e3f0ff 100%);
  min-height: 100vh;
}

.header-section {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 2rem;
  background: linear-gradient(90deg, #e3f0ff 60%, #f5f7fa 100%);
  border-radius: 18px;
  padding: 1.8rem;
  box-shadow: 0 4px 18px rgba(80,120,200,0.10);
}

.section-title {
  color: #2c3e50;
  display: flex;
  align-items: center;
  gap: 0.7rem;
  margin: 0;
  font-size: 2.1rem;
  font-weight: 700;
  letter-spacing: 1px;
}

.section-title i {
  color: #4a89dc;
  font-size: 2.1rem;
}

.section-description {
  color: #7f8c8d;
  font-size: 1rem;
  margin: 0.5rem 0 0 0;
  font-weight: 400;
}

.refresh-btn {
  padding: 0.8rem 1.5rem;
  background: linear-gradient(135deg, #4a89dc, #6dd5ed);
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 1.05rem;
  font-weight: 500;
  transition: all 0.3s;
  box-shadow: 0 2px 8px rgba(74,137,220,0.15);
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.refresh-btn:hover:not(:disabled) {
  background: linear-gradient(135deg, #3a79cc, #5cc5dd);
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(74,137,220,0.25);
}

.refresh-btn:disabled {
  background: #b0c4de;
  cursor: not-allowed;
  transform: none;
}

.upload-section {
  background: white;
  border-radius: 16px;
  padding: 2rem;
  margin-bottom: 2rem;
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
  align-items: center;
}

.file-upload-container {
  width: 70%;
  margin-bottom: 1.5rem;
}

.file-upload-button {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 2rem;
  border: 2px dashed #4a89dc;
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.3s ease;
  width: 100%;
  text-align: center;
  color: #4a89dc;
  font-size: 1.1rem;
  font-weight: 600;
  min-height: 120px;
}

.file-upload-button:hover:not(.disabled) {
  border-color: #5b9dff;
  background-color: #f8faff;
  transform: translateY(-2px);
}

.file-upload-button input[type="file"] {
  display: none;
}

.file-upload-button.disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.progress-container {
  width: 100%;
  height: 10px;
  background-color: #e0e0e0;
  border-radius: 8px;
  margin-top: 1rem;
  overflow: hidden;
  position: relative;
}

.progress-bar {
  height: 100%;
  background: linear-gradient(90deg, #4a89dc, #5b9dff);
  border-radius: 8px;
  transition: width 0.3s ease;
}

.progress-text {
  position: absolute;
  right: 10px;
  top: -18px;
  font-size: 0.8rem;
  color: #4a89dc;
  font-weight: 500;
}

.upload-actions {
  display: flex;
  justify-content: center;
  gap: 1rem;
  width: 100%;
}

.action-btn {
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  gap: 8px;
}

.upload-btn {
  background: linear-gradient(135deg, #4a89dc, #5b9dff);
  color: white;
  box-shadow: 0 4px 6px rgba(74, 137, 220, 0.15);
}

.clear-btn {
  background: linear-gradient(135deg, #8fa6c2, #a1b5d0);
  color: white;
  box-shadow: 0 4px 6px rgba(143, 166, 194, 0.15);
}

.action-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
}

.action-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

/* 加载状态 */
.loading-section {
  text-align: center;
  padding: 2rem;
  color: #4a89dc;
  font-size: 1.2rem;
  font-weight: 500;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
}

.success-message {
  background: linear-gradient(135deg, #27ae60, #2ecc71);
  color: white;
  padding: 1rem 1.5rem;
  border-radius: 8px;
  margin-bottom: 2rem;
  display: flex;
  align-items: center;
  gap: 10px;
  font-weight: 500;
  box-shadow: 0 4px 15px rgba(46, 204, 113, 0.2);
}

.error-text {
  color: #e74c3c;
  font-weight: 500;
}

.uploading-text {
  color: #4a89dc;
  font-weight: 500;
}

/* 集合部分 */
.collections-section {
  margin-bottom: 2rem;
}

.sub-section-title {
  font-size: 1.4rem;
  font-weight: 600;
  color: #2c3e50;
  margin-bottom: 1.5rem;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid #eee;
}

.model-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 1.5rem;
}

.model-card {
  background: white;
  border-radius: 16px;
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.07);
  overflow: hidden;
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  position: relative;
}

.model-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 12px 40px rgba(74, 137, 220, 0.1);
}

.model-card-header {
  padding: 1.5rem;
  background: linear-gradient(135deg, #4a89dc 0%, #5b9dff 100%);
  color: white;
  display: flex;
  align-items: center;
  gap: 1rem;
}

.model-icon {
  background: rgba(255, 255, 255, 0.2);
  width: 48px;
  height: 48px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.5rem;
}

.model-info {
  flex: 1;
}

.model-name {
  margin: 0 0 0.25rem 0;
  font-size: 1.1rem;
  font-weight: 600;
}

.model-type {
  margin: 0;
  font-size: 0.9rem;
  opacity: 0.8;
}

.model-card-body {
  padding: 1.5rem;
}

.model-filename {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #7f8c8d;
  font-size: 0.95rem;
}

.model-filename i {
  color: #4a89dc;
}

/* 文件格式信息 */
.format-info {
  background: white;
  border-radius: 16px;
  padding: 2rem;
  margin-top: 2rem;
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.07);
}

.format-list {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 1rem;
}

.format-item {
  background: linear-gradient(90deg, #f5f7fa 90%, #e3f0ff 100%);
  padding: 1rem;
  border-radius: 10px;
  display: flex;
  align-items: center;
  gap: 10px;
  border: 1px solid rgba(74, 137, 220, 0.1);
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.03);
}

.format-item i {
  color: #4a89dc;
  font-size: 1.2rem;
}

/* 空状态 */
.empty-state {
  grid-column: 1 / -1;
  text-align: center;
  padding: 3rem;
  color: #7f8c8d;
  background: #f9f9f9;
  border-radius: 12px;
  border: 1px dashed #e0e0e0;
}

.empty-state i {
  font-size: 3rem;
  margin-bottom: 1rem;
  color: #bdc3c7;
}

.empty-state h3 {
  font-size: 1.4rem;
  font-weight: 600;
  color: #34495e;
  margin: 0 0 0.5rem 0;
}

.empty-state p {
  color: #7f8c8d;
}

/* 通知样式 */
.notification {
  position: fixed;
  top: 2rem;
  right: 2rem;
  padding: 1rem 1.5rem;
  border-radius: 8px;
  color: white;
  font-weight: 500;
  z-index: 1100;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  min-width: 300px;
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
  animation: slideIn 0.3s ease;
}

@keyframes slideIn {
  from {
    transform: translateX(100%);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
}

.notification.success {
  background: linear-gradient(135deg, #27ae60, #2ecc71);
}

.notification.error {
  background: linear-gradient(135deg, #e74c3c, #c0392b);
}

.notification.warning {
  background: linear-gradient(135deg, #f39c12, #e67e22);
}

/* 响应式设计 */
@media (max-width: 768px) {
  .container {
    padding: 1rem;
  }
  
  .header-section {
    flex-direction: column;
    gap: 1rem;
    align-items: flex-start;
  }
  
  .upload-actions {
    flex-direction: column;
    width: 100%;
  }
  
  .action-btn {
    width: 100%;
    justify-content: center;
  }
  
  .model-grid {
    grid-template-columns: 1fr;
  }
  
  .format-list {
    grid-template-columns: 1fr;
  }
}
</style>
