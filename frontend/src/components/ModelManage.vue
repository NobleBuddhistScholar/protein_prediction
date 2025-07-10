<template>
  <div class="container">
    <div class="header-section">
      <h2 class="section-title">
        <i class="fa-solid fa-database"></i> 模型管理
      </h2>
      <div class="header-actions">
        <button class="model-btn promoter-btn" @click="pullPromoterModel" :disabled="pulling.promoter">
          <i class="fa-solid fa-cloud-download-alt"></i> 
          <span v-if="!pulling.promoter">拉取启动子识别模型</span>
          <span v-else>拉取中...</span>
        </button>
        <button class="model-btn embedding-btn" @click="pullEmbeddingModel" :disabled="pulling.embedding">
          <i class="fa-solid fa-cloud-download-alt"></i> 
          <span v-if="!pulling.embedding">拉取文本向量化模型</span>
          <span v-else>拉取中...</span>
        </button>
        <button class="refresh-btn" @click="loadModels" :disabled="loading">
          <i class="fa-solid fa-sync" :class="{ 'fa-spin': loading }"></i> 刷新
        </button>
      </div>
    </div>

    <!-- 搜索区域 -->
    <div class="search-section">
      <div class="search-container">
        <i class="fa-solid fa-search search-icon"></i>
        <input 
          v-model="searchKeyword" 
          class="search-input" 
          placeholder="搜索模型..." 
        />
      </div>
      <div class="model-count">
        共 {{ filteredModels.length }} 个模型
      </div>
    </div>

    <!-- 加载状态 -->
    <div v-if="loading" class="loading-section">
      <i class="fa-solid fa-spinner fa-spin"></i> 加载中...
    </div>

    <!-- 错误状态 -->
    <div v-if="error" class="error-section">
      <i class="fa-solid fa-exclamation-triangle"></i> {{ error }}
    </div>

    <!-- 模型列表 -->
    <div v-if="!loading && !error" class="model-grid">
      <div 
        v-for="model in filteredModels" 
        :key="model.filename" 
        class="model-card"
      >
        <div class="model-card-header">
          <div class="model-icon">
            <i class="fa-solid fa-brain"></i>
          </div>
          <div class="model-info">
            <h3 class="model-name">{{ model.display_name }}</h3>
            <p class="model-type">{{ model.model_type || '未知类型' }}</p>
          </div>
        </div>
        
        <div class="model-card-body">
          <div class="model-filename">
            <i class="fa-solid fa-file"></i> {{ model.filename }}
          </div>
        </div>
        
        <div class="model-card-actions">
          <button 
            class="action-btn view-btn" 
            @click="viewModelDetails(model.filename)"
            :disabled="actionLoading[model.filename]"
          >
            <i class="fa-solid fa-eye"></i> 详情
          </button>
          <button 
            class="action-btn rename-btn" 
            @click="showRenameDialog(model)"
            :disabled="actionLoading[model.filename]"
          >
            <i class="fa-solid fa-edit"></i> 重命名
          </button>
          <button 
            class="action-btn export-btn" 
            @click="exportModel(model.filename)"
            :disabled="actionLoading[model.filename]"
          >
            <i class="fa-solid fa-download"></i> 导出
          </button>
          <button 
            class="action-btn delete-btn" 
            @click="showDeleteDialog(model)"
            :disabled="actionLoading[model.filename]"
          >
            <i class="fa-solid fa-trash"></i> 删除
          </button>
        </div>
        
        <!-- 加载指示器 -->
        <div v-if="actionLoading[model.filename]" class="action-loading">
          <i class="fa-solid fa-spinner fa-spin"></i>
        </div>
      </div>
      
      <!-- 空状态 -->
      <div v-if="filteredModels.length === 0" class="empty-state">
        <i class="fa-solid fa-folder-open"></i>
        <h3>没有找到模型</h3>
        <p v-if="searchKeyword">尝试修改搜索关键词</p>
        <p v-else>model文件夹中没有可用的模型文件</p>
      </div>
    </div>

    <!-- 模型详情模态框 -->
    <div v-if="showDetailsModal" class="modal-overlay" @click="closeDetailsModal">
      <div class="modal-content details-modal" @click.stop>
        <div class="modal-header">
          <h3><i class="fa-solid fa-info-circle"></i> 模型详情</h3>
          <button class="close-btn" @click="closeDetailsModal">
            <i class="fa-solid fa-times"></i>
          </button>
        </div>
        <div class="modal-body">
          <div v-if="selectedModelDetails" class="details-grid">
            <div class="detail-item">
              <label>文件名:</label>
              <span>{{ selectedModelDetails.filename }}</span>
            </div>
            <div class="detail-item">
              <label>文件大小:</label>
              <span>{{ selectedModelDetails.file_size }}</span>
            </div>
            <div class="detail-item">
              <label>模型类型:</label>
              <span>{{ selectedModelDetails.model_type }}</span>
            </div>
            <div class="detail-item">
              <label>最后修改:</label>
              <span>{{ formatDate(selectedModelDetails.last_modified) }}</span>
            </div>
            <div class="detail-item">
              <label>配置文件:</label>
              <span :class="selectedModelDetails.config_available ? 'status-yes' : 'status-no'">
                {{ selectedModelDetails.config_available ? '存在' : '缺少' }}
              </span>
            </div>
            
            <!-- 详细配置信息 -->
            <div v-if="selectedModelDetails.config_available && selectedModelDetails.config_data" class="config-section">
              <h4>配置详情</h4>
              <div class="config-data">
                <pre>{{ JSON.stringify(selectedModelDetails.config_data, null, 2) }}</pre>
              </div>
            </div>
          </div>
          <div v-else class="loading-details">
            <i class="fa-solid fa-spinner fa-spin"></i> 加载详情中...
          </div>
        </div>
      </div>
    </div>

    <!-- 重命名模态框 -->
    <div v-if="showRenameModal" class="modal-overlay" @click="closeRenameModal">
      <div class="modal-content rename-modal" @click.stop>
        <div class="modal-header">
          <h3><i class="fa-solid fa-edit"></i> 重命名模型</h3>
          <button class="close-btn" @click="closeRenameModal">
            <i class="fa-solid fa-times"></i>
          </button>
        </div>
        <div class="modal-body">
          <div class="form-group">
            <label>当前名称:</label>
            <span class="current-name">{{ selectedModel && selectedModel.filename }}</span>
          </div>
          <div class="form-group">
            <label for="newName">新名称:</label>
            <input 
              id="newName"
              v-model="newModelName" 
              class="rename-input" 
              placeholder="输入新的模型名称"
              @keyup.enter="confirmRename"
            />
            <small class="input-hint">名称将自动添加.pth后缀</small>
          </div>
        </div>
        <div class="modal-footer">
          <button class="btn btn-secondary" @click="closeRenameModal">取消</button>
          <button 
            class="btn btn-primary" 
            @click="confirmRename"
            :disabled="!newModelName.trim() || renaming"
          >
            <i v-if="renaming" class="fa-solid fa-spinner fa-spin"></i>
            {{ renaming ? '重命名中...' : '确认' }}
          </button>
        </div>
      </div>
    </div>

    <!-- 删除确认模态框 -->
    <div v-if="showDeleteModal" class="modal-overlay" @click="closeDeleteModal">
      <div class="modal-content delete-modal" @click.stop>
        <div class="modal-header">
          <h3><i class="fa-solid fa-exclamation-triangle text-danger"></i> 删除模型</h3>
          <button class="close-btn" @click="closeDeleteModal">
            <i class="fa-solid fa-times"></i>
          </button>
        </div>
        <div class="modal-body">
          <p>确定要删除模型 <strong>{{ selectedModel && selectedModel.filename }}</strong> 吗？</p>
          <p class="warning-text">此操作不可撤销，模型文件和配置文件都会被删除。</p>
        </div>
        <div class="modal-footer">
          <button class="btn btn-secondary" @click="closeDeleteModal">取消</button>
          <button 
            class="btn btn-danger" 
            @click="confirmDelete"
            :disabled="deleting"
          >
            <i v-if="deleting" class="fa-solid fa-spinner fa-spin"></i>
            {{ deleting ? '删除中...' : '确认删除' }}
          </button>
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
import { API_BASE_URL } from '../config.js';
export default {
  data() {
    return {
      searchKeyword: '',
      models: [],
      loading: false,
      error: null,
      actionLoading: {},
      
      // 模型拉取状态
      pulling: {
        promoter: false,
        embedding: false
      },
      
      // 模态框状态
      showDetailsModal: false,
      showRenameModal: false,
      showDeleteModal: false,
      
      // 选中的模型
      selectedModel: null,
      selectedModelDetails: null,
      
      // 重命名相关
      newModelName: '',
      renaming: false,
      
      // 删除相关
      deleting: false,
      
      // 通知系统
      notification: {
        show: false,
        message: '',
        type: 'success' // success, error, warning
      }
    };
  },
  
  computed: {
    filteredModels() {
      if (!this.searchKeyword.trim()) {
        return this.models;
      }
      const keyword = this.searchKeyword.toLowerCase().trim();
      return this.models.filter(model => 
        model.filename.toLowerCase().includes(keyword) ||
        model.display_name.toLowerCase().includes(keyword) ||
        (model.model_type && model.model_type.toLowerCase().includes(keyword))
      );
    },
    
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
    this.loadModels();
  },
  
  methods: {
    async loadModels() {
      this.loading = true;
      this.error = null;
      
      try {
        const response = await fetch(`${API_BASE_URL}/commence/models`);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        this.models = data.models || [];
        
        // 重置操作加载状态
        this.actionLoading = {};
        
      } catch (error) {
        console.error('加载模型列表失败:', error);
        this.error = error.message || '加载模型列表失败';
        this.models = [];
      } finally {
        this.loading = false;
      }
    },
    
    async viewModelDetails(filename) {
      this.setActionLoading(filename, true);
      
      try {
        const response = await fetch(`${API_BASE_URL}/model/models/${filename}/details`);
        if (!response.ok) {
          throw new Error(`获取模型详情失败: ${response.statusText}`);
        }
        
        this.selectedModelDetails = await response.json();
        this.showDetailsModal = true;
        
      } catch (error) {
        console.error('获取模型详情失败:', error);
        this.showNotification('获取模型详情失败: ' + error.message, 'error');
      } finally {
        this.setActionLoading(filename, false);
      }
    },
    
    showRenameDialog(model) {
      this.selectedModel = model;
      this.newModelName = model.filename.replace('.pth', '');
      this.showRenameModal = true;
    },
    
    async confirmRename() {
      if (!this.newModelName.trim() || this.renaming) return;
      
      this.renaming = true;
      
      try {
        const response = await fetch(`${API_BASE_URL}/model/models/${this.selectedModel.filename}/rename`, {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            new_name: this.newModelName.trim()
          })
        });
        
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || '重命名失败');
        }
        
        this.showNotification('模型重命名成功', 'success');
        this.closeRenameModal();
        this.loadModels(); // 重新加载模型列表
        
      } catch (error) {
        console.error('重命名模型失败:', error);
        this.showNotification('重命名失败: ' + error.message, 'error');
      } finally {
        this.renaming = false;
      }
    },
    
    showDeleteDialog(model) {
      this.selectedModel = model;
      this.showDeleteModal = true;
    },
    
    async confirmDelete() {
      if (this.deleting) return;
      
      this.deleting = true;
      
      try {
        const response = await fetch(`${API_BASE_URL}/model/models/${this.selectedModel.filename}`, {
          method: 'DELETE'
        });
        
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || '删除失败');
        }
        
        this.showNotification('模型删除成功', 'success');
        this.closeDeleteModal();
        this.loadModels(); // 重新加载模型列表
        
      } catch (error) {
        console.error('删除模型失败:', error);
        this.showNotification('删除失败: ' + error.message, 'error');
      } finally {
        this.deleting = false;
      }
    },
    
    async exportModel(filename) {
      this.setActionLoading(filename, true);
      try {
        const response = await fetch(`${API_BASE_URL}/model/models/${filename}/export`);
        if (!response.ok) {
          throw new Error(`导出失败: ${response.statusText}`);
        }
        // 兼容所有环境的文件名提取
        const contentDisposition = response.headers.get('content-disposition');
        let serverFilename = filename;
        if (contentDisposition && contentDisposition.indexOf('filename=') !== -1) {
          const tmp = contentDisposition.split('filename=')[1];
          serverFilename = tmp ? tmp.replace(/"/g, '') : filename;
        }
        // 创建下载链接
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = serverFilename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        this.showNotification('模型导出成功', 'success');
      } catch (error) {
        console.error('导出模型失败:', error);
        this.showNotification('导出失败: ' + error.message, 'error');
      } finally {
        this.setActionLoading(filename, false);
      }
    },
    
    // 辅助方法
    setActionLoading(filename, loading) {
      this.$set(this.actionLoading, filename, loading);
    },
    
    // 拉取启动子识别模型
    async pullPromoterModel() {
      if (this.pulling.promoter) return;
      
      this.pulling.promoter = true;
      
      try {
        const response = await fetch(`${API_BASE_URL}/model/models/pull/promoter`, {
          method: 'POST'
        });
        
        const data = await response.json();
        
        if (!response.ok) {
          throw new Error(data.error || '拉取启动子识别模型失败');
        }
        
        // 如果有警告信息，使用警告类型通知
        if (data.warning) {
          this.showNotification(data.message, 'warning');
        } else {
          this.showNotification(data.message || '启动子识别模型拉取成功', 'success');
        }
        this.loadModels(); // 刷新模型列表
        
      } catch (error) {
        console.error('拉取启动子识别模型失败:', error);
        this.showNotification('拉取失败: ' + error.message, 'error');
      } finally {
        this.pulling.promoter = false;
      }
    },
    
    // 拉取向量化模型
    async pullEmbeddingModel() {
      if (this.pulling.embedding) return;
      
      this.pulling.embedding = true;
      
      try {
        const response = await fetch(`${API_BASE_URL}/model/models/pull/embedding`, {
          method: 'POST'
        });
        
        const data = await response.json();
        
        if (!response.ok) {
          throw new Error(data.error || '拉取向量化模型失败');
        }
        
        // 如果有警告信息，使用警告类型通知
        if (data.warning) {
          this.showNotification(data.message, 'warning');
        } else {
          this.showNotification(data.message || '向量化模型拉取成功', 'success');
        }
        
      } catch (error) {
        console.error('拉取向量化模型失败:', error);
        this.showNotification('拉取失败: ' + error.message, 'error');
      } finally {
        this.pulling.embedding = false;
      }
    },
    
    closeDetailsModal() {
      this.showDetailsModal = false;
      this.selectedModelDetails = null;
    },
    
    closeRenameModal() {
      this.showRenameModal = false;
      this.selectedModel = null;
      this.newModelName = '';
      this.renaming = false;
    },
    
    closeDeleteModal() {
      this.showDeleteModal = false;
      this.selectedModel = null;
      this.deleting = false;
    },
    
    showNotification(message, type = 'success') {
      this.notification = {
        show: true,
        message,
        type
      };
      
      // 3秒后自动隐藏
      setTimeout(() => {
        this.notification.show = false;
      }, 3000);
    },
    
    formatDate(timestamp) {
      if (!timestamp) return '未知';
      const date = new Date(timestamp * 1000);
      return date.toLocaleString('zh-CN');
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

.header-actions {
  display: flex;
  gap: 1rem;
  align-items: center;
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

.model-btn, .refresh-btn {
  padding: 0.8rem 1.5rem;
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 0.95rem;
  font-weight: 500;
  transition: all 0.3s;
  box-shadow: 0 2px 8px rgba(74,137,220,0.15);
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.refresh-btn {
  background: linear-gradient(135deg, #4a89dc, #6dd5ed);
}

.promoter-btn {
  background: linear-gradient(135deg, #3498db, #2980b9);
}

.embedding-btn {
  background: linear-gradient(135deg, #9b59b6, #8e44ad);
}

.model-btn:hover:not(:disabled), .refresh-btn:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(74,137,220,0.25);
}

.promoter-btn:hover:not(:disabled) {
  background: linear-gradient(135deg, #2980b9, #1c6ea4);
}

.embedding-btn:hover:not(:disabled) {
  background: linear-gradient(135deg, #8e44ad, #7d3c98);
}

.model-btn:disabled, .refresh-btn:disabled {
  opacity: 0.7;
  cursor: not-allowed;
  transform: none;
}

.search-section {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 2rem;
  background: linear-gradient(90deg, #f5f7fa 80%, #e3f0ff 100%);
  border-radius: 12px;
  padding: 1.5rem;
  box-shadow: 0 2px 8px rgba(80,120,200,0.06);
}

.search-container {
  position: relative;
  flex: 1;
  max-width: 400px;
}

.search-icon {
  position: absolute;
  left: 1rem;
  top: 50%;
  transform: translateY(-50%);
  color: #4a89dc;
  font-size: 1rem;
}

.search-input {
  width: 100%;
  padding: 0.75rem 1rem 0.75rem 2.5rem;
  border: 1.5px solid #b0c4de;
  border-radius: 10px;
  font-size: 1.08rem;
  font-weight: 500;
  color: #34495e;
  transition: all 0.3s;
  background: linear-gradient(90deg, #f5f7fa 80%, #e3f0ff 100%);
  box-shadow: 0 2px 8px rgba(80,120,200,0.06);
}

.search-input:focus {
  outline: none;
  border-color: #4a89dc;
  box-shadow: 0 0 0 3px rgba(74,137,220,0.1);
  transform: translateY(-1px);
}

.model-count {
  color: #34495e;
  font-weight: 600;
  font-size: 1.05rem;
  letter-spacing: 0.3px;
}

.loading-section, .error-section {
  text-align: center;
  padding: 3rem;
  background: white;
  border-radius: 18px;
  box-shadow: 0 8px 32px rgba(80,120,200,0.10);
}

.loading-section {
  color: #4a89dc;
  font-size: 1.2rem;
  font-weight: 500;
}

.error-section {
  color: #e74c3c;
  font-size: 1.2rem;
  font-weight: 500;
  background: rgba(231, 76, 60, 0.05);
}

.model-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
  gap: 2rem;
}

.model-card {
  background: white;
  border-radius: 18px;
  box-shadow: 0 8px 32px rgba(80,120,200,0.10);
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
  border: none;
}

.model-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 15px 40px rgba(74,137,220,0.15);
}

.model-card-header {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1.5rem;
  background: linear-gradient(135deg, #2b5694 0%, #3a6cb9 100%);
  color: white;
}

.model-icon {
  width: 50px;
  height: 50px;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.2);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.5rem;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.model-info h3 {
  margin: 0;
  font-size: 1.2rem;
  font-weight: 600;
  letter-spacing: 0.3px;
}

.model-type {
  margin: 0.25rem 0 0 0;
  opacity: 0.9;
  font-size: 0.9rem;
  letter-spacing: 0.2px;
}

.model-card-body {
  padding: 1.8rem;
}

.model-description {
  color: #34495e;
  margin-bottom: 1rem;
  line-height: 1.6;
  min-height: 2.5rem;
  font-size: 1.05rem;
}

.model-filename {
  color: #4a89dc;
  font-size: 0.98rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-family: 'Monaco', 'Courier New', monospace;
  background: rgba(74,137,220,0.1);
  padding: 0.6rem 0.8rem;
  border-radius: 8px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.model-card-actions {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.8rem;
  padding: 0.5rem 1.8rem 1.8rem;
}

.action-btn {
  padding: 0.7rem;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 500;
  font-size: 0.95rem;
  transition: all 0.3s;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 2px 8px rgba(80,120,200,0.08);
  gap: 0.5rem;
}

.view-btn {
  background: linear-gradient(135deg, #74b9ff, #0984e3);
  color: white;
}

.rename-btn {
  background: linear-gradient(135deg, #fdcb6e, #e17055);
  color: white;
}

.export-btn {
  background: linear-gradient(135deg, #00b894, #00a085);
  color: white;
}

.delete-btn {
  background: linear-gradient(135deg, #fd79a8, #e84393);
  color: white;
}

.action-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}

.action-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
}

.action-loading {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(255, 255, 255, 0.9);
  display: flex;
  align-items: center;
  justify-content: center;
  color: #3498db;
  font-size: 1.2rem;
}

.empty-state {
  grid-column: 1 / -1;
  text-align: center;
  padding: 4rem 2rem;
  background: white;
  border-radius: 18px;
  box-shadow: 0 8px 30px rgba(80,120,200,0.1);
  border: 1px solid rgba(80,120,200,0.05);
}

.empty-state i {
  font-size: 4rem;
  color: #4a89dc;
  opacity: 0.4;
  margin-bottom: 1rem;
}

.empty-state h3 {
  color: #2c3e50;
  margin-bottom: 0.5rem;
}

.empty-state p {
  color: #7f8c8d;
}

/* 模态框样式 */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  backdrop-filter: blur(5px);
}

.modal-content {
  background: white;
  border-radius: 16px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
  max-width: 90vw;
  max-height: 90vh;
  overflow-y: auto;
}

.details-modal {
  width: 600px;
}

.rename-modal, .delete-modal {
  width: 450px;
}

.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1.5rem;
  background: linear-gradient(135deg, #4a89dc 0%, #5b9dff 100%);
  color: white;
  border-radius: 16px 16px 0 0;
}

.modal-header h3 {
  margin: 0;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.close-btn {
  background: none;
  border: none;
  color: white;
  font-size: 1.5rem;
  cursor: pointer;
  padding: 0.25rem;
  border-radius: 4px;
  transition: background 0.3s ease;
}

.close-btn:hover {
  background: rgba(255, 255, 255, 0.2);
}

.modal-body {
  padding: 2rem;
}

.details-grid {
  display: grid;
  gap: 1rem;
}

.detail-item {
  display: grid;
  grid-template-columns: 140px 1fr;
  gap: 1rem;
  align-items: start;
  padding: 0.9rem;
  background: linear-gradient(90deg, #f5f7fa 80%, #e3f0ff 100%);
  border-radius: 10px;
  border: 1px solid rgba(80,120,200,0.05);
  box-shadow: 0 2px 8px rgba(80,120,200,0.04);
}

.detail-item label {
  font-weight: 600;
  color: #2c3e50;
  letter-spacing: 0.3px;
}

.detail-item span {
  color: #34495e;
  font-weight: 500;
}

.status-yes {
  color: #27ae60 !important;
  font-weight: 600;
}

.status-no {
  color: #e74c3c !important;
  font-weight: 600;
}

.config-section {
  grid-column: 1 / -1;
  margin-top: 1.5rem;
  background: linear-gradient(90deg, #f8fafd 80%, #f0f7ff 100%);
  border-radius: 12px;
  padding: 1.2rem;
  border: 1px solid rgba(80,120,200,0.08);
  box-shadow: 0 2px 8px rgba(80,120,200,0.04);
}

.config-section h4 {
  color: #2c3e50;
  margin-bottom: 1.2rem;
  font-size: 1.08rem;
  font-weight: 600;
  letter-spacing: 0.3px;
  padding-bottom: 0.6rem;
  border-bottom: 1px solid rgba(80,120,200,0.1);
}

.config-data {
  background: #2d3748;
  color: #e2e8f0;
  padding: 1.2rem;
  border-radius: 8px;
  font-family: 'Monaco', 'Courier New', monospace;
  font-size: 0.9rem;
  line-height: 1.5;
  overflow-x: auto;
  max-height: 300px;
  overflow-y: auto;
  box-shadow: inset 0 0 10px rgba(0,0,0,0.1);
  border: 1px solid #4a5568;
}

.loading-details {
  text-align: center;
  padding: 2.5rem;
  color: #4a89dc;
  font-size: 1.05rem;
  font-weight: 500;
}

.form-group {
  margin-bottom: 1.5rem;
}

.form-group label {
  display: block;
  margin-bottom: 0.6rem;
  font-weight: 600;
  color: #2c3e50;
  font-size: 1.05rem;
  letter-spacing: 0.3px;
}

.current-name {
  color: #34495e;
  font-family: 'Monaco', 'Courier New', monospace;
  background: rgba(74,137,220,0.08);
  padding: 0.7rem 1rem;
  border-radius: 8px;
  display: inline-block;
  border: 1px solid rgba(74,137,220,0.15);
  font-size: 0.95rem;
  font-weight: 500;
}

.rename-input {
  width: 90%;
  padding: 1rem 1rem;
  border: 1.5px solid #b0c4de;
  border-radius: 10px;
  font-size: 1.05rem;
  font-weight: 500;
  color: #34495e;
  transition: all 0.3s;
  background: linear-gradient(90deg, #f5f7fa 80%, #e3f0ff 100%);
  box-shadow: 0 2px 8px rgba(80,120,200,0.06);
}

.rename-input:focus {
  outline: none;
  border-color: #4a89dc;
  box-shadow: 0 0 0 3px rgba(74,137,220,0.1);
  transform: translateY(-1px);
}

.input-hint {
  color: #7f8c8d;
  font-size: 0.8rem;
  margin-top: 0.25rem;
  display: block;
}

.modal-footer {
  display: flex;
  gap: 1rem;
  justify-content: flex-end;
  padding: 1.5rem;
  background: #f8f9fa;
  border-radius: 0 0 16px 16px;
}

.btn {
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.btn-primary {
  background: linear-gradient(135deg, #4a89dc, #5b9dff);
  color: white;
  box-shadow: 0 2px 8px rgba(74,137,220,0.15);
  font-weight: 500;
}

.btn-secondary {
  background: linear-gradient(135deg, #8fa6c2, #a1b5d0);
  color: white;
  box-shadow: 0 2px 8px rgba(143,166,194,0.15);
  font-weight: 500;
}

.btn-danger {
  background: linear-gradient(135deg, #e74c3c, #c0392b);
  color: white;
  box-shadow: 0 2px 8px rgba(231,76,60,0.15);
  font-weight: 500;
}

.btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
}

.warning-text {
  color: #e67e22;
  font-style: italic;
  margin-top: 0.5rem;
}

.text-danger {
  color: #e74c3c;
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
  
  .model-grid {
    grid-template-columns: 1fr;
    gap: 1rem;
  }
  
  .header-section {
    flex-direction: column;
    gap: 1rem;
    text-align: center;
  }
  
  .search-section {
    flex-direction: column;
    gap: 1rem;
  }
  
  .model-card-actions {
    grid-template-columns: 1fr;
  }
  
  .modal-content {
    margin: 1rem;
    width: auto;
  }
}
</style>