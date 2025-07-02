<template>
  <div class="container">
    <div class="header-section">
      <h2 class="section-title">
        <i class="fa-solid fa-database"></i> 模型管理
      </h2>
      <button class="refresh-btn" @click="loadModels" :disabled="loading">
        <i class="fa-solid fa-sync" :class="{ 'fa-spin': loading }"></i> 刷新
      </button>
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
          <div class="model-description">
            {{ model.description || '暂无描述' }}
          </div>
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
export default {
  data() {
    return {
      searchKeyword: '',
      models: [],
      loading: false,
      error: null,
      actionLoading: {},
      
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
        const response = await fetch('http://localhost:5000/models');
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
        const response = await fetch(`http://localhost:5000/models/${filename}/details`);
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
        const response = await fetch(`http://localhost:5000/models/${this.selectedModel.filename}/rename`, {
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
        const response = await fetch(`http://localhost:5000/models/${this.selectedModel.filename}`, {
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
        const response = await fetch(`http://localhost:5000/models/${filename}/export`);
        if (!response.ok) {
          throw new Error(`导出失败: ${response.statusText}`);
        }
        
        // 创建下载链接
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
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
  padding: 2rem;
  font-family: 'Arial', sans-serif;
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  min-height: 100vh;
}

.header-section {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 2rem;
  background: white;
  border-radius: 12px;
  padding: 1.5rem;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

.section-title {
  color: #2c3e50;
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin: 0;
  font-size: 1.8rem;
  font-weight: 600;
}

.section-title i {
  color: #3498db;
  font-size: 2rem;
}

.refresh-btn {
  padding: 0.75rem 1.5rem;
  background: linear-gradient(135deg, #3498db, #2980b9);
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.3s ease;
  box-shadow: 0 2px 10px rgba(52, 152, 219, 0.3);
}

.refresh-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 20px rgba(52, 152, 219, 0.4);
}

.refresh-btn:disabled {
  opacity: 0.7;
  cursor: not-allowed;
  transform: none;
}

.search-section {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 2rem;
  background: white;
  border-radius: 12px;
  padding: 1.5rem;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
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
  color: #7f8c8d;
  font-size: 1rem;
}

.search-input {
  width: 100%;
  padding: 0.75rem 1rem 0.75rem 2.5rem;
  border: 2px solid #ecf0f1;
  border-radius: 25px;
  font-size: 1rem;
  transition: all 0.3s ease;
  background: #f8f9fa;
}

.search-input:focus {
  outline: none;
  border-color: #3498db;
  background: white;
  box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
}

.model-count {
  color: #7f8c8d;
  font-weight: 500;
  font-size: 0.95rem;
}

.loading-section, .error-section {
  text-align: center;
  padding: 3rem;
  background: white;
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

.loading-section {
  color: #3498db;
  font-size: 1.1rem;
}

.error-section {
  color: #e74c3c;
  font-size: 1.1rem;
}

.model-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
  gap: 2rem;
}

.model-card {
  background: white;
  border-radius: 16px;
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.model-card:hover {
  transform: translateY(-8px);
  box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
}

.model-card-header {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1.5rem;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
}

.model-info h3 {
  margin: 0;
  font-size: 1.2rem;
  font-weight: 600;
}

.model-type {
  margin: 0.25rem 0 0 0;
  opacity: 0.9;
  font-size: 0.9rem;
}

.model-card-body {
  padding: 1.5rem;
}

.model-description {
  color: #5a6c7d;
  margin-bottom: 1rem;
  line-height: 1.5;
  min-height: 2.5rem;
}

.model-filename {
  color: #7f8c8d;
  font-size: 0.85rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-family: 'Monaco', 'Courier New', monospace;
  background: #f8f9fa;
  padding: 0.5rem;
  border-radius: 6px;
}

.model-card-actions {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.5rem;
  padding: 0 1.5rem 1.5rem;
}

.action-btn {
  padding: 0.75rem;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 500;
  font-size: 0.9rem;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  justify-content: center;
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
  border-radius: 16px;
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
}

.empty-state i {
  font-size: 4rem;
  color: #bdc3c7;
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
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
  padding: 0.75rem;
  background: #f8f9fa;
  border-radius: 8px;
}

.detail-item label {
  font-weight: 600;
  color: #2c3e50;
}

.detail-item span {
  color: #5a6c7d;
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
  margin-top: 1rem;
}

.config-section h4 {
  color: #2c3e50;
  margin-bottom: 1rem;
}

.config-data {
  background: #2c3e50;
  color: #ecf0f1;
  padding: 1rem;
  border-radius: 8px;
  font-family: 'Monaco', 'Courier New', monospace;
  font-size: 0.85rem;
  overflow-x: auto;
  max-height: 300px;
  overflow-y: auto;
}

.loading-details {
  text-align: center;
  padding: 2rem;
  color: #3498db;
}

.form-group {
  margin-bottom: 1.5rem;
}

.form-group label {
  display: block;
  margin-bottom: 0.5rem;
  font-weight: 600;
  color: #2c3e50;
}

.current-name {
  color: #7f8c8d;
  font-family: 'Monaco', 'Courier New', monospace;
  background: #f8f9fa;
  padding: 0.5rem;
  border-radius: 4px;
  display: inline-block;
}

.rename-input {
  width: 100%;
  padding: 0.75rem;
  border: 2px solid #ecf0f1;
  border-radius: 8px;
  font-size: 1rem;
  transition: border-color 0.3s ease;
}

.rename-input:focus {
  outline: none;
  border-color: #3498db;
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
  background: linear-gradient(135deg, #3498db, #2980b9);
  color: white;
}

.btn-secondary {
  background: #95a5a6;
  color: white;
}

.btn-danger {
  background: linear-gradient(135deg, #e74c3c, #c0392b);
  color: white;
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