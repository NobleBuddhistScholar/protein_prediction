<template>
  <div class="container">
    <div class="header-section">
      <h2 class="section-title">
        <i class="fa-solid fa-file-chart-line"></i> 基因组分析报告
      </h2>
      <button class="refresh-btn" @click="loadReports" :disabled="loading">
        <i class="fa-solid fa-sync" :class="{ 'fa-spin': loading }"></i> 刷新
      </button>
    </div>

    <!-- 搜索区域 -->
    <div class="search-section">
      <div class="search-container">
        <i class="fa-solid fa-search search-icon"></i>
        <input 
          v-model="searchId" 
          class="search-input" 
          placeholder="按基因组ID搜索报告..." 
        />
      </div>
      <div class="report-count">
        共 {{ filteredReports.length }} 个报告
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

    <!-- 报告列表 -->
    <div v-if="!loading && !error" class="report-grid">
      <div 
        v-for="(report, index) in filteredReports" 
        :key="index" 
        class="report-card"
      >
        <div class="report-card-header">
          <div class="report-icon">
            <i class="fa-solid fa-chart-line"></i>
          </div>
          <div class="report-info">
            <h3 class="report-name">{{ report.genome_id }}</h3>
            <p class="report-date">{{ formatDate(report.current_date) }}</p>
          </div>
        </div>
        
        <div class="report-card-body">
          <div class="report-filename">
            <i class="fa-solid fa-file"></i> {{ report.filename }}
          </div>
        </div>
        
        <div class="report-card-actions">
          <button 
            class="action-btn view-btn" 
            @click="showReport(report)"
            :disabled="actionLoading[report.filename]"
          >
            <i class="fa-solid fa-eye"></i> 查看
          </button>
          <button 
            class="action-btn details-btn" 
            @click="viewReportDetails(report.filename)"
            :disabled="actionLoading[report.filename]"
          >
            <i class="fa-solid fa-info-circle"></i> 详情
          </button>
          <button 
            class="action-btn download-btn" 
            @click="downloadReport(report)"
            :disabled="actionLoading[report.filename]"
          >
            <i class="fa-solid fa-download"></i> 下载TXT
          </button>
          <button 
            class="action-btn gff-btn" 
            @click="downloadGFFReport(report)"
            :disabled="actionLoading[report.filename]"
          >
            <i class="fa-solid fa-file-code"></i> 下载GFF
          </button>
          <button 
            class="action-btn delete-btn" 
            @click="showDeleteDialog(report)"
            :disabled="actionLoading[report.filename]"
          >
            <i class="fa-solid fa-trash"></i> 删除
          </button>
        </div>
        
        <!-- 加载指示器 -->
        <div v-if="actionLoading[report.filename]" class="action-loading">
          <i class="fa-solid fa-spinner fa-spin"></i>
        </div>
      </div>
      
      <!-- 空状态 -->
      <div v-if="filteredReports.length === 0" class="empty-state">
        <i class="fa-solid fa-folder-open"></i>
        <h3>没有找到报告</h3>
        <p v-if="searchId">尝试修改搜索关键词</p>
        <p v-else>暂无基因组分析报告</p>
      </div>
    </div>

    <!-- 报告内容模态框 -->
    <div v-if="selectedReport" class="modal-overlay" @click="closeReport">
      <div class="modal-content report-modal" @click.stop>
        <div class="modal-header">
          <h3><i class="fa-solid fa-file-chart-line"></i> 分析报告</h3>
          <button class="close-btn" @click="closeReport">
            <i class="fa-solid fa-times"></i>
          </button>
        </div>
        <div class="modal-body">
          <div v-if="summary" class="report-content" v-html="formattedSummary"></div>
          <div v-else class="loading-details">
            <i class="fa-solid fa-spinner fa-spin"></i> 加载报告中...
          </div>
        </div> 
      </div>
    </div>

    <!-- 报告详情模态框 -->
    <div v-if="showDetailsModal" class="modal-overlay" @click="closeDetailsModal">
      <div class="modal-content details-modal" @click.stop>
        <div class="modal-header">
          <h3><i class="fa-solid fa-info-circle"></i> 报告详情</h3>
          <button class="close-btn" @click="closeDetailsModal">
            <i class="fa-solid fa-times"></i>
          </button>
        </div>
        <div class="modal-body">
          <div v-if="selectedReportDetails" class="details-grid">
            <div class="detail-item">
              <label>文件名:</label>
              <span>{{ selectedReportDetails.filename }}</span>
            </div>
            <div class="detail-item">
              <label>基因组ID:</label>
              <span>{{ selectedReportDetails.genome_id }}</span>
            </div>
            <div class="detail-item">
              <label>分析日期:</label>
              <span>{{ selectedReportDetails.analysis_date }}</span>
            </div>
            <div class="detail-item">
              <label>文件大小:</label>
              <span>{{ selectedReportDetails.file_size }}</span>
            </div>
            <div class="detail-item">
              <label>最后修改:</label>
              <span>{{ formatTimestamp(selectedReportDetails.last_modified) }}</span>
            </div>
          </div>
          <div v-else class="loading-details">
            <i class="fa-solid fa-spinner fa-spin"></i> 加载详情中...
          </div>
        </div>
      </div>
    </div>

    <!-- 删除确认模态框 -->
    <div v-if="showDeleteModal" class="modal-overlay" @click="closeDeleteModal">
      <div class="modal-content delete-modal" @click.stop>
        <div class="modal-header">
          <h3><i class="fa-solid fa-exclamation-triangle text-danger"></i> 删除报告</h3>
          <button class="close-btn" @click="closeDeleteModal">
            <i class="fa-solid fa-times"></i>
          </button>
        </div>
        <div class="modal-body">
          <p>确定要删除报告 <strong>{{ selectedReportForDelete && selectedReportForDelete.filename }}</strong> 吗？</p>
          <p class="warning-text">此操作将同时删除相关的JSON和GFF文件，且不可撤销。</p>
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
            reports: [],
            selectedReport: null,
            summary: null,
            searchId: '',
            loading: false,
            error: null,
            actionLoading: {},
            
            // 模态框状态
            showDetailsModal: false,
            showDeleteModal: false,
            
            // 选中的报告
            selectedReportDetails: null,
            selectedReportForDelete: null,
            
            // 删除相关
            deleting: false,
            
            // 通知系统
            notification: {
                show: false,
                message: '',
                type: 'success'
            }
        };
    },
    
    computed: {
        formattedSummary() {
            if (!this.summary) return '';
            
            // 全局清理Markdown符号
            const cleanedSummary = this.summary.replace(/[*#]/g, '');
            
            const lines = cleanedSummary.split('\n');
            const processed = lines.reduce((acc, line) => {
                line = line.trim();
                const isTableRow = /^\|.*\|$/.test(line);

                if (isTableRow) {
                    acc.tableRows.push(line);
                    acc.inTable = true;
                } else {
                    if (acc.inTable) {
                        const tableHtml = this.processTable(acc.tableRows);
                        acc.htmlArray.push(tableHtml);
                        acc.tableRows = [];
                        acc.inTable = false;
                    }
                    let processedLine = this.processTextLine(line);
                    if (processedLine) acc.htmlArray.push(processedLine);
                }
                return acc;
            }, { inTable: false, tableRows: [], htmlArray: [] });

            if (processed.inTable) {
                const tableHtml = this.processTable(processed.tableRows);
                processed.htmlArray.push(tableHtml);
            }

            let html = processed.htmlArray.join('')
                .replace(/<li>/g, '<ul><li>')
                .replace(/<\/li>/g, '</li></ul>')
                .replace(/<\/ul><ul>/g, '');

            return html;
        },
        
        filteredReports() {
            if (!this.searchId.trim()) {
                return this.reports;
            }
            const keyword = this.searchId.toLowerCase().trim();
            return this.reports.filter(report => 
                report.genome_id.toLowerCase().includes(keyword) ||
                report.filename.toLowerCase().includes(keyword)
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
        this.loadReports();
    },
    
    methods: {
        async loadReports() {
            this.loading = true;
            this.error = null;
            
            try {
                const response = await fetch(`${API_BASE_URL}/getAllSummaries`);
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const data = await response.json();
                this.reports = data.map(filename => {
                    const parts = filename.split('_基因组分析报告_');
                    if (parts.length === 2) {
                        const genome_id = parts[0];
                        const date = parts[1].replace('.txt', '');
                        return {
                            filename,
                            genome_id,
                            current_date: date
                        };
                    }
                    return null;
                }).filter(report => report !== null);
                
                // 重置操作加载状态
                this.actionLoading = {};
                
            } catch (error) {
                console.error('加载报告列表失败:', error);
                this.error = error.message || '加载报告列表失败';
                this.reports = [];
            } finally {
                this.loading = false;
            }
        },
        
        async showReport(report) {
            if (!report || !report.filename) {
                this.showNotification('报告信息不完整或文件名未定义', 'error');
                return;
            }
            
            this.setActionLoading(report.filename, true);
            
            try {
                const { genome_id, current_date } = report;
                const response = await fetch(`${API_BASE_URL}/summary?genome_id=${genome_id}&current_date=${current_date}`);
                
                if (!response.ok) {
                    throw new Error(`获取报告内容失败: ${response.statusText}`);
                }
                
                const data = await response.json();
                this.summary = Object.values(data)[0];
                this.selectedReport = report.filename;
                
            } catch (error) {
                console.error('获取报告内容失败:', error);
                this.showNotification('获取报告内容失败: ' + error.message, 'error');
            } finally {
                this.setActionLoading(report.filename, false);
            }
        },
        
        async viewReportDetails(filename) {
            this.setActionLoading(filename, true);
            
            try {
                const response = await fetch(`${API_BASE_URL}/reports/${filename}/details`);
                if (!response.ok) {
                    throw new Error(`获取报告详情失败: ${response.statusText}`);
                }
                
                this.selectedReportDetails = await response.json();
                this.showDetailsModal = true;
                
            } catch (error) {
                console.error('获取报告详情失败:', error);
                this.showNotification('获取报告详情失败: ' + error.message, 'error');
            } finally {
                this.setActionLoading(filename, false);
            }
        },
        
        async downloadReport(report) {
            this.setActionLoading(report.filename, true);
            
            try {
                const { genome_id, current_date } = report;
                const response = await fetch(`${API_BASE_URL}/summary?genome_id=${genome_id}&current_date=${current_date}`);
                
                if (!response.ok) {
                    throw new Error(`获取报告内容失败: ${response.statusText}`);
                }
                
                const data = await response.json();
                const summary = Object.values(data)[0];
                const element = document.createElement('a');
                element.href = URL.createObjectURL(
                    new Blob([summary.replace(/[*#]/g, '')], 
                    { type: 'text/plain' })
                );
                element.download = report.filename;
                element.click();
                
                this.showNotification('报告下载成功', 'success');
                
            } catch (error) {
                console.error('下载报告失败:', error);
                this.showNotification('下载报告失败: ' + error.message, 'error');
            } finally {
                this.setActionLoading(report.filename, false);
            }
        },
        
        async downloadGFFReport(report) {
            this.setActionLoading(report.filename, true);
            
            try {
                const { genome_id, current_date } = report;
                const gffFileName = `${genome_id}_annotation_${current_date}.gff`;
                const response = await fetch(`${API_BASE_URL}/gff?genome_id=${genome_id}&current_date=${current_date}`);
                
                if (!response.ok) {
                    throw new Error(`获取GFF报告失败: ${response.statusText}`);
                }
                
                const data = await response.json();
                const gffContent = data[gffFileName];
                const formattedContent = gffContent.replace(/(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.+)/g, '$1\t$2\t$3\t$4\t$5\t$6\t$7\t$8\t$9');
                const blob = new Blob([formattedContent], { type: 'text/plain' });
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = gffFileName;
                a.click();
                window.URL.revokeObjectURL(url);
                
                this.showNotification('GFF报告下载成功', 'success');
                
            } catch (error) {
                console.error('下载GFF报告失败:', error);
                this.showNotification('下载GFF报告失败: ' + error.message, 'error');
            } finally {
                this.setActionLoading(report.filename, false);
            }
        },
        
        showDeleteDialog(report) {
            this.selectedReportForDelete = report;
            this.showDeleteModal = true;
        },
        
        async confirmDelete() {
            if (this.deleting) return;
            
            this.deleting = true;
            
            try {
                const response = await fetch(`${API_BASE_URL}/reports/${this.selectedReportForDelete.filename}`, {
                    method: 'DELETE'
                });
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || '删除失败');
                }
                
                this.showNotification('报告删除成功', 'success');
                this.closeDeleteModal();
                this.loadReports(); // 重新加载报告列表
                
            } catch (error) {
                console.error('删除报告失败:', error);
                this.showNotification('删除失败: ' + error.message, 'error');
            } finally {
                this.deleting = false;
            }
        },
        
        processTextLine(line) {
            // 行级二次清理
            line = line.replace(/[*#]/g, '');
            
            if (/^\d+\.\s/.test(line)) {
                return `<h3>${line.replace(/^\d+\.\s/, '')}</h3>`;
            }
            if (/^- /.test(line)) {
                return `<li>${line.replace(/^- /, '')}</li>`;
            }
            return line ? `<p>${line}</p>` : '';
        },
        
        processTable(rows) {
            if (rows.length < 2) return '';
            
            // 表头清理
            const headers = rows[0]
                .split('|')
                .slice(1, -1)
                .map(h => h.trim().replace(/[*#]/g, ''));

            // 表格内容清理
            const dataRows = rows.slice(2).map(row => 
                row.split('|')
                   .slice(1, -1)
                   .map(c => c.trim().replace(/[*#]/g, ''))
            );

            let html = '<table class="summary-table"><thead><tr>';
            headers.forEach(header => {
                html += `<th>${header}</th>`;
            });
            html += '</tr></thead><tbody>';

            dataRows.forEach(cells => {
                html += '<tr>';
                cells.forEach(cell => {
                    html += `<td>${cell}</td>`;
                });
                html += '</tr>';
            });

            return html + '</tbody></table>';
        },
        
        downloadSummary() {
            if (this.summary) {
                const element = document.createElement('a');
                element.href = URL.createObjectURL(
                    new Blob([this.summary.replace(/[*#]/g, '')], 
                    { type: 'text/plain' })
                );
                element.download = this.selectedReport;
                element.click();
                this.showNotification('报告下载成功', 'success');
            }
        },
        
        // 辅助方法
        setActionLoading(filename, loading) {
            this.$set(this.actionLoading, filename, loading);
        },
        
        closeReport() {
            this.selectedReport = null;
            this.summary = null;
        },
        
        closeDetailsModal() {
            this.showDetailsModal = false;
            this.selectedReportDetails = null;
        },
        
        closeDeleteModal() {
            this.showDeleteModal = false;
            this.selectedReportForDelete = null;
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
        
        formatDate(dateStr) {
            if (!dateStr) return '未知';
            // 如果是 YYYY-MM-DD 格式，转换为更友好的格式
            const date = new Date(dateStr);
            if (isNaN(date.getTime())) {
                return dateStr; // 如果解析失败，返回原字符串
            }
            return date.toLocaleDateString('zh-CN', {
                year: 'numeric',
                month: 'long',
                day: 'numeric'
            });
        },
        
        formatTimestamp(timestamp) {
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

.report-count {
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

.report-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
  gap: 2rem;
}

.report-card {
  background: white;
  border-radius: 18px;
  box-shadow: 0 8px 32px rgba(80,120,200,0.10);
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
  border: none;
}

.report-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 15px 40px rgba(74,137,220,0.15);
}

.report-card-header {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1.5rem;
  background: linear-gradient(135deg, #2b5694 0%, #3a6cb9 100%);
  color: white;
}

.report-icon {
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

.report-info h3 {
  margin: 0;
  font-size: 1.2rem;
  font-weight: 600;
}

.report-date {
  margin: 0.25rem 0 0 0;
  opacity: 0.9;
  font-size: 0.9rem;
}

.report-card-body {
  padding: 1.5rem;
}

.report-filename {
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

.report-card-actions {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.5rem;
  padding: 0 1.5rem 1.5rem;
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
  background: linear-gradient(135deg, #4a89dc, #5b9dff);
  color: white;
}

.details-btn {
  background: linear-gradient(135deg, #74b9ff, #0984e3);
  color: white;
}

.download-btn {
  background: linear-gradient(135deg, #00b894, #00a085);
  color: white;
}

.gff-btn {
  background: linear-gradient(135deg, #4a89dc, #6dd5ed);
  color: white;
}

.delete-btn {
  background: linear-gradient(135deg, #fd79a8, #e84393);
  color: white;
  grid-column: 1 / -1;
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
  color: #e67e22;
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

.report-modal {
  width: 800px;
  max-height: 85vh;
}

.details-modal {
  width: 600px;
}

.delete-modal {
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
  max-height: 60vh;
  overflow-y: auto;
}

.report-content {
  line-height: 1.6;
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

.loading-details {
  text-align: center;
  padding: 2rem;
  color: #e67e22;
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
  background: linear-gradient(135deg, #e67e22, #d35400);
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

/* 报告内容样式 */
.report-content h3 {
  color: #4a89dc;
  margin: 2rem 0 1.5rem;
  padding-bottom: 0.8rem;
  border-bottom: 2px solid rgba(74,137,220,0.15);
  font-size: 1.5rem;
  font-weight: 600;
  letter-spacing: 0.3px;
}

.report-content ul {
  margin: 1.5rem 0;
  padding-left: 2rem;
  list-style: none;
}

.report-content li {
  position: relative;
  margin: 1rem 0;
  color: #34495e;
  font-size: 1.1rem;
  line-height: 1.5;
}

.report-content li::before {
  content: "▪";
  color: #4a89dc;
  position: absolute;
  left: -1.5rem;
  font-size: 1rem;
}

.report-content p {
  color: #34495e;
  line-height: 1.9;
  margin: 1rem 0;
  text-align: justify;
  font-size: 1.1rem;
}

.summary-table {
  width: 100%;
  border-collapse: collapse;
  margin: 2rem 0;
  font-size: 0.9rem;
}

.summary-table th,
.summary-table td {
  border: 1px solid #ddd;
  padding: 15px;
  text-align: left;
}

.summary-table th {
  background-color: #4a89dc;
  color: white;
  font-weight: 600;
}

.summary-table tr:nth-child(even) {
  background-color: #f8f9fa;
}

.summary-table tr:hover {
  background-color: #f1f4f7;
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
  background: linear-gradient(135deg, #4a89dc, #6dd5ed);
}

/* 响应式设计 */
@media (max-width: 768px) {
  .container {
    padding: 1rem;
  }
  
  .report-grid {
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
  
  .report-card-actions {
    grid-template-columns: 1fr;
  }
  
  .modal-content {
    margin: 1rem;
    width: auto;
  }
  
  .report-modal {
    width: auto;
    max-width: 95vw;
  }
}
</style>
