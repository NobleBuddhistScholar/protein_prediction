<template>
  <div class="container">
    <!-- 头部部分 -->
    <div class="header-section">
      <h2 class="section-title">
        <i class="fa-solid fa-microscope"></i> 基因组注释
      </h2>
      <button class="refresh-btn" @click="refreshPage" :disabled="isUploading">
        <i class="fa-solid fa-sync" :class="{ 'fa-spin': isUploading }"></i> 刷新
      </button>
    </div>

    <!-- 上传区域（无基因数据时显示） -->
    <div v-if="!geneData" class="upload-section">
      <h3 class="section-subtitle">
        <i class="fa-solid fa-dna"></i> 上传基因组文件进行分析
      </h3>
      <div class="file-upload-container">
        <div class="file-dropzone" 
             :class="{ 'disabled': isUploading, 'active-dropzone': isDragging }" 
             @dragenter.prevent="isDragging = true" 
             @dragleave.prevent="isDragging = false"
             @dragover.prevent
             @drop.prevent="handleFileDrop">
          <label class="file-upload-button" :class="{ 'disabled': isUploading }">
            <input type="file" ref="fileInput" @change="handleFileUpload" accept=".fasta" :disabled="isUploading" />
            <span v-if="!isUploading && !uploadError">
              <i class="fa-solid fa-cloud-upload-alt"></i> {{ fileName || '选择FASTA文件' }}
            </span>
            <span v-if="isUploading" class="uploading-text">
              <i class="fa-solid fa-spinner fa-spin"></i> 正在分析中...
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
        
        <div class="model-select-container">
          <label class="model-select-label">选择分析模型：</label>
          <select v-model="selectedModelFile" class="model-select" :disabled="isUploading">
            <option v-for="option in modelOptions" :key="option.value" :value="option.value">
              {{ option.label }}
            </option>
          </select>
        </div>
        
        <div class="upload-actions">
          <button class="action-btn analyze-btn" @click="handleAnalyzeFile" :disabled="!fileName || isUploading">
            <i class="fa-solid fa-play"></i> 开始分析
          </button>
          <button class="action-btn clear-btn" @click="clearFileSelection" :disabled="isUploading">
            <i class="fa-solid fa-times"></i> 清除
          </button>
        </div>
      </div>
    </div>

    <div v-if="geneData">
      <h3 class="gene-visualization-title">
        <i class="fa-solid fa-dna"></i> 基因组图谱
      </h3>
      <div class="search-button-container">
        <!-- 新增搜索框 -->
        <input
          type="text"
          v-model="searchKeyword"
          placeholder="按蛋白类型搜索"
          @input="filterGenes"
        />
        <!-- 控制按钮 -->
        <div class="control-buttons">
          <button @click="showReportModal" :disabled="isUploading || !currentReport">
            <i class="fa-solid fa-table-list"></i> 查看txt报告
          </button>
          <button @click="openGffReportModal">
            <i class="fa-solid fa-file-alt"></i> 查看GFF报告
          </button>
          <button @click="clearDetection">
            <i class="fa-solid fa-power-off"></i> 关闭预测
          </button>
        </div>
      </div>

      <div class="gene-visualization-wrapper">
        <!-- 刻度尺容器 -->
        <div class="ruler-container" ref="rulerContainer" @mousemove="handleRulerHover" @mouseleave="handleRulerLeave">
          <canvas ref="rulerCanvas" class="ruler-canvas"></canvas>
          <!-- 基因总长度条 -->
          <div
            class="gene-length-bar"
            :style="{ width: getScaledWidth(geneData.metadata.length) + 'px' }"
          ></div>
          <!-- 新增：显示刻度提示 -->
          <div
            v-if="showRulerTooltip"
            class="ruler-tooltip"
            :style="{ left: rulerTooltipPos + 'px' }"
          >
            {{ rulerTooltipText }}
          </div>
          <!-- 新增：显示垂直虚线 -->
          <div
            v-if="showRulerLine"
            class="ruler-line"
            :style="{ left: rulerTooltipPos + 'px' }"
          ></div>

          <!-- 基因段显示区域 -->
          <div class="gene-rows-container" ref="geneRowsContainer" @scroll="handleScroll">
            <!-- 修改这里：使用filteredGeneRows而不是geneRows -->
            <div v-for="(row, index) in filteredGeneRows" :key="index" class="gene-row">
              <div
                v-for="gene in row"
                :key="gene.location"
                :style="{
                  left: getScaledPosition(getLeftPosition(gene)) + 'px',
                  width: getScaledWidth(getWidth(gene)) + 'px',
                  backgroundColor: getColor(gene.type),
                  zIndex: hoveredGene === gene.location ? 10 : 1,
                }"
                class="gene-item"
                @click="showDetails(gene)"
                @mouseenter="handleGeneHover(gene)"
                @mouseleave="handleGeneLeave(gene)"
              >
                <span class="gene-type-label">
                  {{ getGeneTypeLabel(gene.type) }}
                </span>
                <div v-if="hoveredGene === gene.location" class="gene-tooltip">
                  <div>{{ gene.type }}</div>
                  <div>位置: {{ gene.location }}</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- 基因组详细信息模态框 -->
        <div v-if="showGeneDetailsModal" class="modal">
          <div class="detail-modal-content">
            <span class="detail-close" @click="closeGeneDetailsModal">&times;</span>
            <h3 class="detail-modal-title">基因详细信息</h3>
            <div class="detail-gene-details">
              <div class="detail-detail-row">
                <div class="detail-detail-label">基因类型:</div>
                <div class="detail-detail-value">{{ currentGeneDetails.type }}</div>
              </div>
              <div class="detail-detail-row">
                <div class="detail-detail-label">位置:</div>
                <div class="detail-detail-value">{{ currentGeneDetails.location }}</div>
              </div>
              <div class="detail-detail-row">
                <div class="detail-detail-label">长度:</div>
                <div class="detail-detail-value">{{ getWidth(currentGeneDetails) }}</div>
              </div>
              <div class="detail-detail-row">
                <div class="detail-detail-label">置信度:</div>
                <div class="detail-detail-value">{{ currentGeneDetails.qualifiers.confidence }}</div>
              </div>
            </div>
          </div>
        </div>
        <!-- TXT报告模态框 -->
        <div v-if="showTxtReportModal" class="modal">
          <div class="modal-content">
            <div class="modal-header">
              <h3 class="modal-title">TXT 分析报告</h3>
              <button class="modal-download" @click="downloadTxtReport">
                <i class="fa-solid fa-download"></i> 下载
              </button>
              <span class="close" @click="closeTxtReportModal">&times;</span>
            </div>
            <div class="report-content" v-html="formatContent(txtReportContent)"></div>
          </div>
        </div>
        <!-- GFF报告模态框 -->
        <div v-if="showGffReportModal" class="modal">
          <div class="modal-content">
            <div class="modal-header">
              <h3 class="modal-title">GFF报告</h3>
              <button class="modal-download" @click="downloadGffReport">
                <i class="fa-solid fa-download"></i> 下载
              </button>
              <span class="close" @click="closeGffReportModal">&times;</span>
            </div>
            <div class="gff-modal-content" v-html="gffReport"></div>
          </div>
        </div>
        
        <!-- AI流式分析报告对话框（非弹窗，固定在基因图谱下方） -->
        <div class="ai-report-panel">
          <div class="ai-report-panel-header">
            <div class="ai-report-title">
              <i class="fa-solid fa-robot"></i> AI分析报告
            </div>
            <div class="ai-report-controls">
              <button v-if="!currentReport" @click="handleStreamSummary" :disabled="isUploading || !geneData" class="ai-report-generate">
                <i class="fa-solid fa-wand-magic-sparkles" style="color: white;"></i> 生成分析报告
              </button>
              <button v-else @click="handleStreamSummary" :disabled="isUploading || !geneData" class="ai-report-regenerate">
                <i class="fa-solid fa-rotate" style="color: white;"></i> 重新生成
              </button>
              <label class="rag-switch">
                <span class="rag-label">知识库</span>
                <div class="switch-container">
                  <input type="checkbox" v-model="useRAG">
                  <span class="slider"></span>
                </div>
              </label>
            </div>
          </div>
          <div class="ai-report-panel-body" ref="aiReportPanelBody">
            <div v-if="isUploading && !currentReport" class="ai-report-loading">
              <i class="fa-solid fa-spinner fa-spin"></i> 正在生成AI分析报告...
            </div>
            <div v-else-if="uploadError" class="ai-report-error">
              <i class="fa-solid fa-exclamation-triangle"></i> {{ uploadError }}
            </div>
            <div v-else-if="currentReport" v-html="formatContent(currentReport)"></div>
            <div v-else class="ai-report-empty">
              <div v-if="geneData" class="ai-report-empty-message">
                <i class="fa-solid fa-lightbulb"></i>
                <div>
                  <div class="empty-title">基因组已上传成功</div>
                  <div class="empty-desc">点击上方"生成分析报告"按钮获取详细的AI分析。</div>
                </div>
              </div>
              <div v-else class="ai-report-empty-message">
                <i class="fa-solid fa-file-import"></i>
                <div>
                  <div class="empty-title">还未上传基因组</div>
                  <div class="empty-desc">请先上传基因组数据后再生成分析报告。</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { API_BASE_URL } from '../config.js';
export default {
  data() {
    return {
      // 新增状态存储报告信息
      reportInfo: {
        genome_id: '',
        report_date: ''
      },
      // 是否使用知识库增强生成
      useRAG: true,
      geneData: null,
      currentReport: null,
      savedGeneData: null,
      showLocationTooltip: false,
      locationTooltip: '',
      geneRows: [],
      showGeneDetailsModal: false,
      currentGeneDetails: {},
      scaleFactor: 0.035, // 减小比例因子，使可视化更紧凑
      // 新增状态
      fileName: '',
      isUploading: false,
      progress: 0,
      progressInterval: null,
      // 新增状态
      hoveredGene: null,
      showRulerTooltip: false,
      rulerTooltipPos: 0,
      rulerTooltipText: '',
      showRulerLine: true, // 新增：显示垂直虚线状态
      // 新增搜索关键词状态
      searchKeyword: '',
      // 新增过滤后的基因行状态
      filteredGeneRows: [],
      // 新增GFF报告相关状态
      showGffReportModal: false,
      gffReport: null,
      isGeneratingGff: false, // 新增状态，用于显示生成GFF文件的加载提示
      uploadError: null, // 初始化uploadError
      // 新增模型选择相关状态
      selectedModelFile: 'HyperFusionCortex_v1.pth', // 默认选择的模型
      modelOptions: [], // 模型选项列表
      // 新增：TXT报告模态框显示状态
      showTxtReportModal: false,
      // 新增：TXT报告内容
      txtReportContent: '',
      // 拖拽上传相关状态
      isDragging: false,
      uploadSuccess: false
    };
  },
  computed: {
    formattedReport() {
      if (!this.currentReport) return '';
      return this.formatContent(this.currentReport);
    }
  },
  watch: {
    geneData() {
      this.$nextTick(() => {
        this.drawRuler();
      });
    },
    // AI报告流式内容变化时自动滚动到底部
    currentReport() {
      this.$nextTick(() => {
        this.scrollToBottom();
      });
    }
  },
  mounted() {
    if (this.geneData) {
      this.drawRuler();
    }
    // 加载模型列表
    this.loadModelList();
  },
  methods: {
    // 刷新整个页面的方法
    refreshPage() {
      //this.isUploading = true;
      window.location.reload();
    },
    // 新增：加载模型列表
    async loadModelList() {
      try {
        const response = await fetch(`${API_BASE_URL}/models`);
        if (!response.ok) throw new Error('获取模型列表失败');
        
        const data = await response.json();
        this.modelOptions = data.models.map(model => ({
          value: model.filename,
          label: model.display_name + (model.description ? ` (${model.description})` : '')
        }));
        
        // 如果当前选择的模型不在列表中，选择第一个可用模型
        if (this.modelOptions.length > 0 && 
            !this.modelOptions.find(option => option.value === this.selectedModelFile)) {
          this.selectedModelFile = this.modelOptions[0].value;
        }
      } catch (error) {
        console.error('加载模型列表失败:', error);
        // 提供默认选项
        this.modelOptions = [
          { value: 'HyperFusionCortex_v1.pth', label: 'HyperFusionCortex v1' }
        ];
      }
    },

    drawRuler() {
      const canvas = this.$refs.rulerCanvas;
      if (!canvas || !this.geneData) return;

      const ctx = canvas.getContext('2d');
      const totalLength = this.geneData.metadata.length;
      const scaledWidth = this.getScaledWidth(totalLength);

      // 设置canvas物理尺寸
      canvas.width = scaledWidth;
      canvas.height = 30; // 增加高度空间

      // 设置canvas显示尺寸（CSS单位）
      canvas.style.width = scaledWidth + 'px';
      canvas.style.height = '30px';

      // 清空画布
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // 设置绘制样式
      ctx.strokeStyle = '#666';
      ctx.fillStyle = '#666';
      ctx.font = '10px Arial';
      ctx.textAlign = 'center';

      // 每1000bp一个主刻度
      for (let i = 0; i <= totalLength; i += 1000) {
        const pos = this.getScaledPosition(i);
        // 绘制主刻度线
        ctx.beginPath();
        ctx.moveTo(pos, 0);
        ctx.lineTo(pos, 15);
        ctx.stroke();

        // 绘制刻度标签
        ctx.fillText(`${i / 1000}k`, pos, 25);
      }

      // 每500bp一个次刻度
      ctx.strokeStyle = '#999';
      for (let i = 0; i <= totalLength; i += 500) {
        if (i % 1000 === 0) continue; // 跳过主刻度
        const pos = this.getScaledPosition(i);
        ctx.beginPath();
        ctx.moveTo(pos, 5);
        ctx.lineTo(pos, 10);
        ctx.stroke();
      }
    },

    // 处理刻度尺悬停
    handleRulerHover(event) {
      const ruler = event.currentTarget;
      const rect = ruler.getBoundingClientRect();
      const pos = event.clientX - rect.left;
      const position = Math.round(pos / this.scaleFactor);

      this.rulerTooltipPos = pos;
      this.rulerTooltipText = position;
      this.showRulerTooltip = true;
      this.showRulerLine = true; // 显示垂直虚线
    },
    // 处理刻度尺鼠标离开
    handleRulerLeave() {
      this.showRulerTooltip = false;
      this.showRulerLine = false; // 隐藏垂直虚线
    },
    // 处理基因段悬停
    handleGeneHover(gene) {
      this.hoveredGene = gene.location;
    },

    // 处理基因段离开
    handleGeneLeave(gene) {
      this.hoveredGene = null;
    },
    // 处理滚动同步
    handleScroll() {
      const container = this.$refs.geneRowsContainer;
      const rulerContainer = this.$refs.rulerContainer;
      if (container && rulerContainer) {
        rulerContainer.scrollLeft = container.scrollLeft;
      }
    },

    handleFileUpload(event) {
      const file = event.target.files[0];
      if (!file) return;

      // 只记录文件信息，不进行上传和分析
      this.fileName = file.name;
      this.uploadError = null;
      this.uploadSuccess = false;
      
      // 文件已选择，但尚未上传
      console.log(`文件 ${this.fileName} 已选择，点击"开始分析"按钮开始处理`);
    },
    
    async analyzeFile() {
      if (!this.fileName || this.isUploading || !this.$refs.fileInput.files[0]) return;
      
      const file = this.$refs.fileInput.files[0];
      this.isUploading = true;
      this.progress = 0;

      // 检查 rulerCanvas 是否存在
      if (this.$refs.rulerCanvas) {
        // 绘制刻度尺
        this.drawRuler();
      }

      // 模度条更新
      this.progressInterval = setInterval(() => {
        if (this.progress < 90) {
          this.progress += Math.random() * 2;
        }
      }, 1000);

      const formData = new FormData();
      formData.append('file', file);
      formData.append('model_file', this.selectedModelFile); // 添加选择的模型文件参数

      try {
        const response = await fetch(`${API_BASE_URL}/upload`, {
          method: 'POST',
          body: formData
        });

        if (!response.ok) throw new Error('上传失败');

        const data = await response.json();
        this.geneData = data.results;
        this.currentReport = data.summary;
        this.processGeneRows();

        // 新增：从响应头获取报告文件名
    // 修改：直接从响应数据中获取文件名（假设字段为 filename ）
        // 新增：从响应头获取报告文件名
        let filename = '';
        if (data.summary && data.summary.filename) {
          filename = data.summary.filename;
        } else if (data.summary) {
          filename = Object.keys(data.summary)[0];
        } else if (data.filename) {
          filename = data.filename;
        } else {
          filename = '';
        }

        // 解析基因组ID和日期
        const [genome_id, report_date] = this.parseFilename(filename);

        this.reportInfo = {
          genome_id,
          report_date
        };

        // 完成进度
        this.progress = 100;
        setTimeout(async () => {
          this.isUploading = false;
          clearInterval(this.progressInterval);
          this.uploadSuccess = true;
          
          // 3秒后自动隐藏成功消息
          setTimeout(() => {
            this.uploadSuccess = false;
          }, 3000);
          
          // 基因组处理完成后，不再自动生成AI报告
          // 用户需要点击"生成分析报告"按钮才会生成
          console.log('基因组处理完成，等待用户手动生成AI分析报告...');
        }, 500);
      } catch (error) {
        console.error('Error:', error);
        this.uploadError = '文件处理失败: ' + error.message;
        this.isUploading = false;
        clearInterval(this.progressInterval);
        alert('文件处理失败: ' + error.message);
      }
    },

    // 新增方法：解析文件名
    parseFilename(filename) {
      const pattern = /(.+)_基因组分析报告_(\d{4}-\d{2}-\d{2})\.txt/;
      const match = filename.match(pattern);
      return match ? [match[1], match[2]] : ['', ''];
    },

    // 获取TXT报告并显示为模态框（数据驱动）
    async showReportModal() {
      // 如果正在上传或生成AI报告，或者报告尚未生成，则不执行操作
      if (this.isUploading || !this.currentReport) {
        console.log('报告尚未生成或正在生成中，请稍候...');
        return;
      }
      
      if (!this.geneData) {
        alert('没有基因组数据，无法显示报告');
        return;
      }
      try {
        const genome_id = this.geneData.metadata.genome_id;
        const current_date = new Date().toISOString().split('T')[0];
        const url = `${API_BASE_URL}/summary?genome_id=${genome_id}&current_date=${current_date}`;
        const response = await fetch(url);
        if (!response.ok) throw new Error('报告获取失败');
        const reportData = await response.json();
        const reportContent = reportData[Object.keys(reportData)[0]];
        this.txtReportContent = reportContent;
        this.showTxtReportModal = true;
        // 解析文件名
        this.reportInfo = {
          genome_id,
          report_date: current_date
        };
      } catch (error) {
        console.error('TXT报告加载失败:', error);
        this.txtReportContent = `无法加载报告内容：${error.message}`;
        this.showTxtReportModal = true;
      }
    },
    // 关闭TXT报告模态框
    closeTxtReportModal() {
      this.showTxtReportModal = false;
    },
    // 下载TXT报告
    downloadTxtReport() {
      if (!this.txtReportContent) return;
      const { genome_id, report_date } = this.reportInfo;
      const filename = `${genome_id || 'report'}_基因组分析报告_${report_date || ''}.txt`;
      const blob = new Blob([this.txtReportContent], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    },
    async openGffReportModal() {
    // 如果正在上传或者生成AI报告，则不执行操作
    if (this.isUploading) {
      console.log('正在生成报告，请稍候...');
      return;
    }
      
    this.showGffReportModal = true;
    this.isGeneratingGff = true;
    try {
      // 从geneData中获取genome_id
      const genome_id = this.geneData.metadata.genome_id;

      // 生成当前日期
      const current_date = new Date().toISOString().split('T')[0];

      // 发出生成GFF报告的请求
      const generateResponse = await fetch(
        `${API_BASE_URL}/generate_gff?genome_id=${genome_id}&current_date=${current_date}`
      );
      if (!generateResponse.ok) {
        throw new Error('生成GFF报告失败');
      }

      // 获取GFF报告
      const gffResponse = await fetch(
        `${API_BASE_URL}/gff?genome_id=${genome_id}&current_date=${current_date}`
      );
      if (!gffResponse.ok) {
        throw new Error('获取GFF报告失败');
      }

      const gffData = await gffResponse.json();
      this.gffReport = gffData[Object.keys(gffData)[0]];
    } catch (error) {
      console.error('GFF报告处理失败:', error);
      this.gffReport = `无法加载GFF报告内容：${error.message}`;
    } finally {
      this.isGeneratingGff = false;
    }
  },

    // 新增方法：关闭GFF报告模态框
    closeGffReportModal() {
      this.showGffReportModal = false;
    },

    clearDetection() {
      this.geneData = null;
      this.currentReport = null;
      this.savedGeneData = null;
      this.$refs.fileInput.value = '';
      this.fileName = ''; // 重置文件名
      this.uploadError = null; // 重置上传错误信息
      this.gffReport = null; // 重置GFF报告
    },

    // 清除旧的closeModal方法
    

    formatContent(content) {
      const cleaned = content.replace(/[*#]/g, '');
      const lines = cleaned.split('\n');
      let html = [];
      let tableRows = [];
      let inTable = false;

      lines.forEach(line => {
        line = line.trim();
        const isTableRow = /^\|.+?\|$/.test(line);

        if (isTableRow) {
          if (!inTable) inTable = true;
          tableRows.push(line);
        } else {
          if (inTable) {
            html.push(this.processTable(tableRows));
            tableRows = [];
            inTable = false;
          }
          html.push(this.processTextLine(line));
        }
      });

      return html.join('');
    },


    processTextLine(line) {
      line = line.replace(/[*#]/g, '');
      if (/^\d+\.\s/.test(line)) return `<h3>${line.replace(/^\d+\.\s/, '')}</h3>`;
      if (/^- /.test(line)) return `<li>${line.replace(/^- /, '')}</li>`;
      return line ? `<p>${line}</p>` : '';
    },

    processTable(rows) {
      if (rows.length < 2) return '';
      const headers = rows[0].split('|').slice(1, -1).map(h => h.trim());
      const dataRows = rows.slice(2);

      let html = '<table class="summary-table"><thead><tr>';
      headers.forEach(h => html += `<th>${h}</th>`);
      html += '</tr></thead><tbody>';

      dataRows.forEach(row => {
        const cells = row.split('|').slice(1, -1).map(c => c.trim());
        html += '<tr>';
        cells.forEach(c => html += `<td>${c}</td>`);
        html += '</tr>';
      });

      return html + '</tbody></table>';
    },

    getColor(type) {
      const colors = {
        // 核心非结构蛋白 (暖色系)
        NSP1: '#FF6F61',  // 珊瑚红
        NSP2: '#FFA630',  // 日落橙
        NSP3: '#FFD166',  // 琥珀黄
        NSP4: '#EF476F',  // 覆盆子红
        
        // 酶相关蛋白 (绿色系)
        NSP5: '#F8F3E6',  // 翡翠绿
        NSP6: '#88D18A',  // 青苔绿
        NSP7: '#B5E48C',  // 嫩草绿
        NSP8: '#4CAF50',  // 森林绿
        
        // 复制相关蛋白 (紫色系)
        NSP9: '#F5EBE0',  // 紫水晶
        NSP10: '#7B68EE', // 中紫罗兰
        NSP11: '#C77DFF', // 薰衣草紫
        NSP12: '#5E548E', // 深紫灰
        
        // 核酸处理蛋白 (大地色系)
        NSP13: '#F0EFEB',  // 陶土棕
        NSP14: '#BC6C25',  // 铜橙色
        NSP15: '#E76F51',  // 赤陶色
        NSP16: '#cccccc',  // 沙金色
        
        // 结构蛋白 (浅中性色)
        membrane_protein: '#9D4EDD',  // 珍珠白
        envelope_protein: '#D4A373',  // 羊皮纸色
        nucleocapsid_protein: '#06D6A0', // 丝绸灰
      };
      return colors[type] || '#E9C46A'; // 默认浅灰色
    },
    showDetails(gene) {
      this.currentGeneDetails = gene;
      this.showGeneDetailsModal = true;
    },

    showLocation(position) {
      this.showLocationTooltip = true;
      this.locationTooltip = position;
    },

    hideLocation() {
      this.showLocationTooltip = false;
    },

    processGeneRows() {
      this.geneRows = [];
      const uniqueGenes = {};
      this.geneData.features.forEach(gene => {
        if (!uniqueGenes[gene.location]) {
          uniqueGenes[gene.location] = gene;
          let placed = false;
          for (let i = 0; i < this.geneRows.length; i++) {
            if (!this.hasOverlap(this.geneRows[i], gene)) {
              this.geneRows[i].push(gene);
              placed = true;
              break;
            }
          }
          if (!placed) {
            this.geneRows.push([gene]);
          }
        }
      });
      // 填充基因行的逻辑
      this.filterGenes();
    },

    // 修改filterGenes方法
    filterGenes() {
      if (!this.searchKeyword) {
        this.filteredGeneRows = JSON.parse(JSON.stringify(this.geneRows)); // 深拷贝原始数据
        return;
      }

      const keyword = this.searchKeyword.toLowerCase().trim();
      this.filteredGeneRows = this.geneRows
        .map(row => row.filter(gene => gene.type.toLowerCase().includes(keyword)))
        .filter(row => row.length > 0); // 只保留有基因的行
    },

    hasOverlap(row, gene) {
      const [start, end] = gene.location.split('..').map(Number);
      return row.some(item => {
        const [itemStart, itemEnd] = item.location.split('..').map(Number);
        return (start < itemEnd && end > itemStart);
      });
    },

    getScaledWidth(length) {
      return length * this.scaleFactor;
    },

    getScaledPosition(position) {
      return position * this.scaleFactor;
    },

    getLeftPosition(gene) {
      return parseInt(gene.location.split('..')[0]);
    },

    getWidth(gene) {
      const [start, end] = gene.location.split('..').map(Number);
      return end - start;
    },
    // 新增方法，用于关闭基因详细信息模态框
    closeGeneDetailsModal() {
      this.showGeneDetailsModal = false;
    },
    // 新增方法：根据基因类型返回显示标签
    getGeneTypeLabel(type) {
      switch (type) {
        case 'membrane_protein':
          return 'M';
        case 'spike_protein':
          return 'S';
        case 'envelope_protein':
          return 'E';
        case 'nucleocapsid_protein':
          return 'N';
        default:
          return type;
      }
    },
        // 新增下载GFF报告的方法
    downloadGffReport() {
      if (this.gffReport) {
        const { genome_id, report_date } = this.reportInfo;
        const filename = `${genome_id || 'gff'}_annotation_${report_date || ''}.gff`;
        const blob = new Blob([this.gffReport], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
      }
    },
    async streamSummary(genome_id, genome_data) {
      this.isUploading = true;
      this.currentReport = '';
      this.uploadError = null;
      try {
        // 将 useRAG 参数添加到请求中
        const postData = { 
          genome_id, 
          genome_data,
          use_rag: this.useRAG  // 添加是否使用知识库增强的参数
        };
        console.log(`生成AI报告，知识库增强: ${this.useRAG ? '开启' : '关闭'}`);
        const response = await fetch(`${API_BASE_URL}/stream_summary`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(postData)
        });
        if (!response.ok) throw new Error(`服务器响应错误: ${response.status} ${response.statusText}`);
        if (!response.body) throw new Error('流式接口无响应体');
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let done = false;
        this.currentReport = '';
        while (!done) {
          const { value, done: streamDone } = await reader.read();
          done = streamDone;
          let chunk = '';
          if (value) {
            chunk = decoder.decode(value, { stream: !done });
            if (chunk.startsWith('[ERROR]')) {
              throw new Error(chunk.substring(7));
            }
            // 追加内容并强制刷新
            this.currentReport += chunk;
            await this.$nextTick();
            this.scrollToBottom();
          }
        }
      } catch (error) {
        this.uploadError = 'AI分析失败: ' + error.message;
        this.currentReport = '生成报告时出错: ' + error.message;
      } finally {
        this.isUploading = false;
        await this.$nextTick();
        this.scrollToBottom();
      }
    },
    async handleStreamSummary() {
      if (!this.geneData) {
        console.error('没有基因数据，无法生成报告');
        this.uploadError = '请先上传基因组数据';
        return;
      }
      
      // 清空旧的报告内容
      this.currentReport = '';
      this.uploadError = null;
      
      try {
        console.log('开始生成AI分析报告...');
        // 确保genome_id存在
        const genome_id = this.geneData.metadata.genome_id || 'unknown';
        
        // 调用流式分析
        await this.streamSummary(genome_id, this.geneData);
      } catch (e) {
        console.error('处理流式分析时出错:', e);
        this.uploadError = 'AI分析失败: ' + e.message;
        this.currentReport = '生成报告失败: ' + e.message;
      }
    },
    
    // 自动滚动到底部的方法
    scrollToBottom() {
      const panel = this.$refs.aiReportPanelBody;
      if (panel) {
        console.log('滚动到底部, 高度:', panel.scrollHeight);
        panel.scrollTop = panel.scrollHeight;
      }
    },

    // 处理文件拖放
    handleFileDrop(event) {
      event.preventDefault();
      this.isDragging = false;
      
      // 获取拖放的文件
      const files = event.dataTransfer.files;
      if (!files || files.length === 0) return;
      
      const file = files[0];
      
      // 检查文件类型是否为FASTA格式
      if (!file.name.toLowerCase().endsWith('.fasta')) {
        this.uploadError = "请上传FASTA格式文件";
        setTimeout(() => {
          this.uploadError = null;
        }, 3000);
        return;
      }
      
      // 手动设置文件到input元素
      const dataTransfer = new DataTransfer();
      dataTransfer.items.add(file);
      this.$refs.fileInput.files = dataTransfer.files;
      
      // 仅调用文件选择处理函数，不会立即分析
      this.handleFileUpload({ target: { files: dataTransfer.files } });
    },
    
    // 清除文件选择
    clearFileSelection() {
      this.fileName = '';
      this.uploadError = null;
      this.uploadSuccess = false;
      this.progress = 0;
      
      // 清除文件输入
      if (this.$refs.fileInput) {
        this.$refs.fileInput.value = null;
      }
    },
    
    // 处理分析文件
    handleAnalyzeFile() {
      if (!this.fileName || this.isUploading) return;
      
      // 调用analyzeFile方法开始分析
      this.analyzeFile();
    },
  }
};
</script>

<style scoped>
/* 上传区样式 */
.upload-section {
  background: white;
  border-radius: 16px;
  padding: 2rem;
  margin: 2rem auto;
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
  max-width: 1200px;
  border: 1px solid rgba(74, 137, 220, 0.15);
}

/* 欢迎信息样式 */
.welcome-message {
  text-align: center;
  max-width: 800px;
  margin: 4rem auto 2rem;
  padding: 2rem;
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  border-radius: 16px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
  animation: fadeIn 0.8s ease-out;
}

.welcome-icon {
  font-size: 4rem;
  color: #4a89dc;
  margin-bottom: 1.5rem;
}

.welcome-message h2 {
  color: #2c3e50;
  margin-bottom: 1rem;
  font-size: 2rem;
}

.welcome-message p {
  color: #34495e;
  font-size: 1.2rem;
  line-height: 1.6;
  max-width: 600px;
  margin: 0 auto;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* 容器样式 */
.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2.5rem 1.5rem 2rem 1.5rem;
  font-family: 'Segoe UI', 'Arial', sans-serif;
  background: linear-gradient(120deg, #f5f7fa 0%, #e3f0ff 100%);
  min-height: 100vh;
}

/* 新增刻度尺样式 */
.ruler-canvas {
  display: block;
  margin-bottom: -10px;
  background: linear-gradient(90deg, #f5f7fa 80%, #e3f0ff 100%);
  border-radius: 10px;
  box-shadow: 0 2px 8px rgba(80,120,200,0.06);
}


.gene-length-bar {
  height: 6px;
  background: linear-gradient(90deg, #0f2a84, #a6d9ef);
  border-radius: 3px;
  margin-top: 26px; /* 与canvas高度对齐 */
  box-shadow: 0 2px 8px rgba(74,137,220,0.15);
}


/* 基因类型标签样式 */
.gene-type-label {
  text-shadow: 1px 1px 2px rgba(0, 87, 255, 0.8);
  font-size: 12px;
  padding: 2px;
  user-select: none;
}


/* 调整基因项悬停效果 */
.gene-item:hover {
  transform: scale(1.05);
  box-shadow: 0 3px 6px rgba(0, 0, 0, 0.2);
  z-index: 15 !important;
}

/* 其他样式保持不变... */
.header-section {
display: flex;
align-items: center;
justify-content: space-between;
margin-bottom: 2rem;
flex-wrap: wrap;
gap: 1.2rem;
padding: 1.5rem 1.8rem;
background: linear-gradient(90deg, #e3f0ff 60%, #f5f7fa 100%);
border-radius: 18px;
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

.section-subtitle {
  font-size: 1.4rem;
  font-weight: 600;
  color: #2c3e50;
  margin-bottom: 1.5rem;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid #eee;
  display: flex;
  align-items: center;
  gap: 0.7rem;
}

.section-subtitle i {
  color: #4a89dc;
}

.file-upload-container {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  width: 70%;
  margin: 0 auto 1.5rem;
}

.file-dropzone {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 2rem;
  border: 2px dashed #4a89dc;
  border-radius: 12px;
  transition: all 0.3s ease;
  width: 100%;
  text-align: center;
  color: #4a89dc;
  background-color: rgba(74, 137, 220, 0.03);
}

.file-dropzone:hover {
  border-color: #5b9dff;
  background-color: rgba(74, 137, 220, 0.08);
  transform: translateY(-2px);
}

.file-dropzone.active-dropzone {
  border-color: #4a89dc;
  background-color: rgba(74, 137, 220, 0.1);
  box-shadow: 0 0 0 3px rgba(74, 137, 220, 0.2);
}

.file-upload-button {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1.2rem;
  color: #4a89dc;
  cursor: pointer;
  transition: all 0.3s;
  font-weight: 600;
  font-size: 1.1rem;
  text-align: center;
  position: relative;
  min-height: 120px;
  width: 100%;
}

.file-upload-button:hover:not(.disabled) {
  color: #3a79cc;
}

.file-upload-button.disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.file-upload-button input[type="file"] {
  position: absolute;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  opacity: 0;
  cursor: pointer;
}

.uploading-text {
display: flex;
align-items: center;
gap: 0.5rem;
}

.progress-container {
  width: 100%;
  height: 10px;
  background-color: #e0e0e0;
  border-radius: 8px;
  margin: 1rem 0;
  overflow: hidden;
  position: relative;
}

.progress-bar {
  height: 100%;
  background: linear-gradient(90deg, #4a89dc, #5b9dff);
  border-radius: 8px;
  transition: width 0.3s ease;
  box-shadow: 0 1px 3px rgba(74, 137, 220, 0.3);
}

.progress-text {
  position: absolute;
  right: 10px;
  top: -18px;
  font-size: 0.8rem;
  color: #4a89dc;
  font-weight: 600;
}

/* 图标样式 */
@keyframes spin {
0% { transform: rotate(0deg); }
100% { transform: rotate(360deg); }
}

.icon-loading {
display: inline-block;
animation: spin 1s linear infinite;
}
/* 整体容器样式 */
.container {
max-width: 1200px;
margin: 0 auto;
padding: 20px;
font-family: 'Arial', sans-serif;
}

/* 控制按钮样式 */
.control-buttons {
margin: 1rem 3px;
display: flex;
gap: 1.2rem;
}

.control-buttons button {
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

.control-buttons button:hover {
background: linear-gradient(135deg, #3a79cc, #5cc5dd);
transform: translateY(-1px);
box-shadow: 0 4px 12px rgba(74,137,220,0.25);
}

/* 新增：按钮禁用状态样式 */
.control-buttons button:disabled {
  background: linear-gradient(135deg, #b3c9e6, #c5e2eb);
  color: #e0e0e0;
  cursor: not-allowed;
  box-shadow: none;
  transform: none;
  opacity: 0.8;
}

.control-buttons button:disabled:hover {
  background: linear-gradient(135deg, #b3c9e6, #c5e2eb);
  transform: none;
  box-shadow: none;
}

/* 基因可视化包装器 */
.gene-visualization-wrapper {
border: none;
border-radius: 18px;
padding: 1.8rem;
background-color: white;
margin-top: 1.5rem;
overflow-y: hidden;
box-shadow: 0 8px 32px rgba(80,120,200,0.10);
}
/* 基因总长度条样式 */
.gene-length-bar {
height: 24px;
background-color: #e0e0e0;
position: relative;
margin-bottom: 10px;
border-radius: 5px;
cursor: crosshair;
}


/* 基因行容器样式 - 可滚动区域 */
.gene-rows-container {
position: relative;
overflow-x: auto;
overflow-y: hidden;
height: auto;
max-height: 400px;
border: none;
border-radius: 12px;
background: linear-gradient(90deg, #f5f7fa 80%, #e3f0ff 100%);
padding: 15px 0;
box-shadow: 0 2px 8px rgba(80,120,200,0.06);
}

/* 基因行样式 */
.gene-row {
position: relative;
height: 30px;
margin-bottom: 5px;
}

/* 基因项样式 */
.gene-item {
position: absolute;
height: 25px;
display: flex;
align-items: center;
justify-content: center;
cursor: pointer;
border-radius: 4px;
box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
color: white;
font-size: 12px;
font-weight: bold;
transition: transform 0.2s, box-shadow 0.2s;
}


/* 模态框样式 */
.modal {
  position: fixed;
  z-index: 9999;
  left: 0;
  top: 0;
  width: 100vw;
  height: 100vh;
  background: rgba(40, 40, 80, 0.35);
  display: flex;
  align-items: center;
  justify-content: center;
  backdrop-filter: blur(2px);
}

.modal-content {
background: white;
padding: 2rem;
border-radius: 16px;
width: 80%;
max-width: 700px;
max-height: 80vh;
overflow-y: auto;
position: relative;
box-shadow: 0 8px 30px rgba(80,120,200,0.15);
}

.modal-title {
color: #2c3e50;
margin: 0;
font-size: 1.3rem;
font-weight: 600;
letter-spacing: 0.3px;
}

.close {
font-size: 1.8rem;
cursor: pointer;
color: #7f8c8d;
transition: color 0.3s;
}

.close:hover {
color: #4a89dc;
}

/* 基因详细信息样式 */
.gene-details {
display: flex;
flex-direction: column;
gap: 0.8rem;
}

.detail-row {
display: flex;
align-items: center;
}

.detail-label {
font-weight: bold;
color: #2c3e50;
min-width: 120px;
}

.detail-value {
color: #34495e;
padding-left: 10px;
}

/* 表格样式 */
.summary-table {
width: 100%;
border-collapse: collapse;
margin: 1.5rem 0;
font-size: 0.9rem;
}

.summary-table th,
.summary-table td {
border: 1px solid #ddd;
padding: 12px;
text-align: left;
}

.summary-table th {
background-color: #4a89dc;
color: white;
font-weight: 600;
}

.summary-table tr:nth-child(even) {
background-color: #f2f2f2;
}

.summary-table tr:hover {
background-color: #e6f2ff;
}

/* 新增样式 */
.ruler-container {
overflow-x: auto;
position: relative;
padding-bottom: 5px;
margin-bottom: 5px;
}

.gene-ruler {
position: relative;
height: 30px;
min-width: 100%;
}


.ruler-tooltip {
position: absolute;
top: 48px;
transform: translateX(-50%);
background: linear-gradient(135deg, #4a89dc, #6dd5ed);
color: white;
padding: 6px 12px;
border-radius: 8px;
font-size: 13px;
font-weight: 500;
z-index: 10;
pointer-events: none;
white-space: nowrap;
box-shadow: 0 4px 12px rgba(74,137,220,0.25);
}

/*垂直虚线*/
.ruler-line {
  position: absolute;
  width: 2px;
  top: 0px;
  bottom: -10px;
  background-color: rgba(74, 137, 220, 0.7);
  background-image: linear-gradient(to bottom, rgba(74, 137, 220, 0.7) 50%, transparent 50%);
  background-size: 1px 4px;
  z-index: 1;
  pointer-events: none;
}

.gene-tooltip {
  position: absolute;
  right: 100%; /* 改为右侧100%，定位到基因段左侧 */
  left: auto;  /* 取消左侧定位 */
  top: 50%;
  transform: translateY(-50%);
  margin-right: 8px; /* 改为右侧边距 */
  margin-left: 0;    /* 取消左侧边距 */
  background-color: rgba(80, 170, 229, 0.9);
  color: white;
  padding: 6px 8px;
  border-radius: 6px;
  font-size: 12px;
  min-width: 120px;
  text-align: left;
  z-index: 2000;
  pointer-events: none;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
  white-space: nowrap;
}

/* 调整小箭头方向 */
.gene-tooltip::before {
  content: '';
  position: absolute;
  left: 100%;  /* 箭头现在在右侧 */
  right: auto;
  top: 50%;
  transform: translateY(-50%);
  border-width: 5px;
  border-style: solid;
  border-color: transparent transparent transparent rgba(80, 170, 229, 0.9); /* 箭头方向调整 */
}

.gene-tooltip div {
margin: 2px 0;
}

/* 增强版搜索框样式 */
input[type="text"] {
  padding: 0.75rem 1rem 0.75rem 2.5rem;
  border: 1.5px solid #b0c4de;
  border-radius: 10px;
  font-size: 1.05rem;
  width: 350px;
  background: linear-gradient(90deg, #f5f7fa 80%, #e3f0ff 100%);
  box-shadow: 0 2px 8px rgba(80,120,200,0.06);
  color: #34495e;
  font-weight: 500;
  transition: all 0.3s;
}

/* 搜索框聚焦样式 */
input[type="text"]:focus {
  outline: none;
  border-color: #4a89dc;
  box-shadow: 0 0 0 3px rgba(74,137,220,0.1);
  transform: translateY(-1px);
}

/* 添加搜索图标 */
input[type="text"] {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' fill='%234a89dc' viewBox='0 0 16 16'%3E%3Cpath d='M11.742 10.344a6.5 6.5 0 1 0-1.397 1.398h-.001c.03.04.062.078.098.115l3.85 3.85a1 1 0 0 0 1.415-1.414l-3.85-3.85a1.007 1.007 0 0 0-.115-.1zM12 6.5a5.5 5.5 0 1 1-11 0 5.5 5.5 0 0 1 11 0z'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: 15px center;
  padding-left: 45px;
  background-size: 16px;
}
/* 新增搜索框和按钮容器样式 */
.search-button-container {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

/* 新增标题样式 */
.gene-visualization-title {
  color: #2c3e50;
  font-family: 'Segoe UI', 'Arial', sans-serif;
  font-size: 1.8rem;
  font-weight: 700;
  margin-top: 2.2rem;
  margin-bottom: 1.2rem;
  padding-top: 1.2rem;
  border-top: 2px solid #e3f0ff;
  letter-spacing: 0.5px;
  display: flex;
  align-items: center;
}

.gene-visualization-title i {
  margin-right: 12px;
  color: #4a89dc;
  font-size: 36px;
}

.gff-header {
  font-size: 1.5rem;
  font-weight: bold;
  margin-bottom: 0.5rem;
}

.gff-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-bottom: 0.3rem;
}


.gff-modal-content {
  width: auto;
  max-width: 90%;
  font-size: 1.1rem;
  padding: 2rem;
  border-radius: 12px;
  background-color: white;
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
  max-height: 80vh;
  overflow-y: auto;
  position: relative;
}

/* 新增：控制GFF标题和下载按钮的容器样式 */
.gff-title-button-container {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1rem;
}

/* 调整下载按钮的样式，使其与标题在同一行显示更美观 */
.download-gff-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 0.75rem 1.5rem;
  background-color: #4a89dc;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.3s;
  font-weight: 500;
  margin-bottom: 0; /* 移除原有的底部外边距 */
}

.download-gff-button:hover {
  background-color: #3b7dd8;
  transform: translateY(-1px);
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.download-gff-button i {
  margin-right: 0.5rem;
}

/* 新增模型选择样式 */
.model-select-container {
  display: flex;
  flex-direction: column;
  width: 100%;
  padding: 0.5rem 0;
}

.model-select-label {
  margin-bottom: 0.8rem;
  font-weight: 600;
  color: #2c3e50;
  font-size: 1.05rem;
}

.model-select {
  padding: 0.8rem 1rem;
  border: 2px solid rgba(74, 137, 220, 0.3);
  border-radius: 10px;
  font-size: 1rem;
  background: white;
  cursor: pointer;
  transition: all 0.2s;
  color: #2c3e50;
  font-weight: 500;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.model-select:focus {
  border-color: #4a89dc;
  box-shadow: 0 0 0 3px rgba(74, 137, 220, 0.15);
  outline: none;
}

.model-select:disabled {
  background-color: #f5f7fa;
  border-color: #e0e0e0;
  cursor: not-allowed;
}

/* 新增AI流式报告样式 */
.ai-report-modal {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 2000;
  animation: fadeIn 0.4s;
}

.ai-report-content {
  background: white;
  padding: 2rem;
  border-radius: 12px;
  width: 90%;
  max-width: 800px;
  max-height: 80vh;
  overflow-y: auto;
  position: relative;
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
}

.ai-report-title {
  color: #2c3e50;
  margin-bottom: 1.5rem;
  font-size: 1.8rem;
  display: flex;
  align-items: center;
}

.ai-report-title i {
  margin-right: 0.5rem;
  color: #4a89dc;
  font-size: 1.2rem;
}

.ai-report-stream {
  font-size: 0.9rem;
  color: #34495e;
  line-height: 1.6;
}

.ai-report-loading {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100px;
  font-size: 1rem;
  color: #7f8c8d;
}

/* 淡入动画 */
@keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}

/* 新增AI报告面板样式 */
.ai-report-panel {
  background: linear-gradient(120deg, #ffffff 60%, #f5f7fa 100%);
  border-radius: 18px;
  box-shadow: 0 8px 32px rgba(80,120,200,0.10);
  margin-top: 2rem;
  padding: 2rem;
  position: relative;
  overflow: hidden;
}

.ai-report-panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1.5rem;
  font-size: 1.4rem;
  color: #2c3e50;
  padding-bottom: 1rem;
  border-bottom: 2px solid #e3f0ff;
  font-weight: 600;
  letter-spacing: 0.3px;
}

.ai-report-panel-header i {
  color: #4a89dc;
  margin-right: 0.8rem;
  font-size: 1.5rem;
}

.ai-report-regenerate {
  background: linear-gradient(135deg, #4a89dc, #6dd5ed);
  color: white;
  border: none;
  border-radius: 8px;
  padding: 0.6rem 1.2rem;
  font-size: 1rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  box-shadow: 0 2px 8px rgba(74,137,220,0.15);
}

.ai-report-regenerate:hover:not(:disabled) {
  background: linear-gradient(135deg, #3a79cc, #4a8def);
  transform: translateY(-2px);
  box-shadow: 0 6px 12px rgba(74,137,220,0.25);
}

.ai-report-regenerate:disabled {
  background: #b0c4de;
  cursor: not-allowed;
}

.ai-report-generate {
  background: linear-gradient(135deg, #4a89dc, #6dd5ed);
  color: white;
  border: none;
  border-radius: 8px;
  padding: 0.6rem 1.2rem;
  font-size: 1rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  box-shadow: 0 2px 8px rgba(52, 152, 219, 0.15);
}

.ai-report-generate:hover:not(:disabled) {
  background: linear-gradient(135deg, #2980b9, #1c6ea4);
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(52, 152, 219, 0.25);
}

.ai-report-generate:disabled {
  background: #b0c4de;
  cursor: not-allowed;
}

.ai-report-panel-body {
  max-height: 400px;
  overflow-y: auto;
  font-size: 1rem;
  color: #34495e;
  line-height: 1.6;
  padding: 1.2rem;
  border-radius: 12px;
  background: linear-gradient(90deg, #f5f7fa 80%, #e3f0ff 100%);
  box-shadow: 0 2px 8px rgba(80,120,200,0.06);
}

.ai-report-loading {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
  color: #666;
  gap: 10px;
  font-size: 1rem;
}

.ai-report-error {
  color: #e74c3c;
  padding: 15px;
  background-color: #fadbd8;
  border-radius: 4px;
  margin: 10px 0;
}

.ai-report-empty {
  color: #7f8c8d;
  text-align: center;
  padding: 20px;
  font-style: italic;
  background-color: #f5f7fa;
  border-radius: 8px;
  margin: 10px 0;
  padding: 20px 10px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 150px;
}

.ai-report-empty-message {
  display: flex;
  align-items: center;
  gap: 15px;
  padding: 15px 20px;
  background-color: #f8f9fa;
  border-radius: 8px;
  border-left: 4px solid #3498db;
  max-width: 450px;
}

.ai-report-empty-message i {
  font-size: 2rem;
  color: #3498db;
}

.empty-title {
  font-weight: 600;
  font-size: 1.1rem;
  color: #2c3e50;
  margin-bottom: 5px;
}

.empty-desc {
  font-size: 0.9rem;
  color: #7f8c8d;
}

/* 强化滚动条样式 */
.ai-report-panel_body::-webkit-scrollbar {
  width: 8px;
}

.ai-report-panel_body::-webkit-scrollbar-thumb {
  background-color: #4a89dc;
  border-radius: 4px;
}

.ai-report-panel_body::-webkit-scrollbar-track {
  background-color: #f1f1f1;
}

/* 增强报告内容格式 */
.ai-report-panel_body h3 {
  margin-top: 12px;
  margin-bottom: 8px;
  color: #2c3e50;
  font-weight: 600;
  font-size: 1.1rem;
}

.ai-report-panel_body p {
  margin-bottom: 8px;
  line-height: 1.6;
}

.ai-report-panel_body li {
  margin-bottom: 4px;
  list-style-position: inside;
}

/* 表格样式优化 */
.ai-report-panel_body .summary-table {
  margin: 15px 0;
  width: 100%;
  border-collapse: collapse;
}

.ai-report-panel_body .summary-table th {
  background-color: #4a89dc;
  color: white;
  padding: 8px;
  text-align: left;
  font-weight: normal;
}

.ai-report-panel_body .summary-table td {
  padding: 6px 8px;
  border: 1px solid #ddd;
}

/* 模态框头部统一布局 */
.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 2px solid #e3f0ff;
  padding-bottom: 1rem;
  margin-bottom: 1.5rem;
}
.modal-title {
  margin: 0;
  font-size: 1.4rem;
  color: #2c3e50;
  font-weight: 600;
  letter-spacing: 0.3px;
}
.modal-download {
  background: linear-gradient(135deg, #4a89dc, #6dd5ed);
  color: #fff;
  border: none;
  border-radius: 8px;
  padding: 0.6rem 1.2rem;
  font-size: 1rem;
  font-weight: 500;
  cursor: pointer;
  margin-right: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  transition: all 0.3s;
  box-shadow: 0 2px 8px rgba(74,137,220,0.15);
}
.modal-download:hover {
  background: linear-gradient(135deg, #3a79cc, #5cc5dd);
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(74,137,220,0.25);
}
.close {
  font-size: 1.8rem;
  color: #7f8c8d;
  cursor: pointer;
  margin-left: 1rem;
  transition: all 0.3s;
}

.ai-report-panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.ai-report-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
}

.ai-report-controls {
  display: flex;
  align-items: center;
  gap: 15px;
}

/* 知识库增强开关按钮样式 */
.rag-switch {
  position: relative;
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  background: linear-gradient(135deg, #4a89dc, #6dd5ed);
  padding: 6px 10px;
  border-radius: 4px;
  border: 1px solid #e2e8f0;
}

.switch-container {
  position: relative;
  display: inline-block;
  width: 36px;
  height: 20px;
}

.rag-switch input {
  opacity: 0;
  width: 0;
  height: 0;
}

.slider {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: #d9cbe1;
  border-radius: 10px;
  transition: .3s;
}

.slider:before {
  position: absolute;
  content: "";
  height: 16px;
  width: 16px;
  left: 2px;
  bottom: 2px;
  background-color: white;
  border-radius: 50%;
  transition: .3s;
}

input:checked + .slider {
  background-color: #3498db;
}

input:focus + .slider {
  box-shadow: 0 0 1px #3498db;
}

input:checked + .slider:before {
  transform: translateX(16px);
}

.rag-label {
  font-size: 0.85em;
  font-weight: 500;
  color: #f8f2f2;
  user-select: none;
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
  background: linear-gradient(135deg, #4a89dc, #6dd5ed);
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(74,137,220,0.25);
}

.refresh-btn:disabled {
  background: #b0c4de;
  cursor: not-allowed;
  transform: none;
}

/* 新增：按钮禁用状态样式 */
.action-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
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
  font-size: 1rem;
}

.analyze-btn {
  background: linear-gradient(135deg, #4a89dc, #5b9dff);
  color: white;
  box-shadow: 0 4px 6px rgba(74, 137, 220, 0.15);
}

.analyze-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 6px 12px rgba(74, 137, 220, 0.25);
  background: linear-gradient(135deg, #3a79cc, #4a8def);
}

.clear-btn {
  background: linear-gradient(135deg, #8fa6c2, #a1b5d0);
  color: white;
  box-shadow: 0 4px 6px rgba(143, 166, 194, 0.15);
}

.clear-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 6px 12px rgba(143, 166, 194, 0.25);
}

.format-info {
  margin-top: 1rem;
  display: flex;
  justify-content: center;
}

.format-badge {
  background: linear-gradient(90deg, #f5f7fa 90%, #e3f0ff 100%);
  padding: 0.7rem 1.2rem;
  border-radius: 20px;
  display: flex;
  align-items: center;
  gap: 8px;
  border: 1px solid rgba(74, 137, 220, 0.1);
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
  color: #4a89dc;
  font-weight: 500;
}

.format-badge i {
  color: #4a89dc;
}

.success-message {
  background: linear-gradient(135deg, #27ae60, #2ecc71);
  color: white;
  padding: 1rem 1.5rem;
  border-radius: 8px;
  margin-top: 1.5rem;
  display: flex;
  align-items: center;
  gap: 10px;
  font-weight: 500;
  box-shadow: 0 4px 15px rgba(46, 204, 113, 0.2);
  animation: fadeIn 0.5s ease-out;
}

.uploading-text {
  color: #4a89dc;
  font-weight: 500;
}

.error-text {
  color: #e74c3c;
  font-weight: 500;
}
/* 特定的基因卡片样式 */
.detail-modal-content {
  background: linear-gradient(120deg, #f5f7fa 60%, #e3f0ff 100%);
  border-radius: 22px;
  box-shadow: 0 8px 40px 0 rgba(80,120,200,0.18), 0 1.5px 8px 0 rgba(123,104,238,0.10);
  padding: 2.5rem 2.8rem 2.2rem 2.8rem;
  min-width: 380px;
  max-width: 95vw;
  min-height: 260px;
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  animation: modal-pop 0.25s cubic-bezier(.4,2,.6,1) 1;
}

.detail-close {
  position: absolute;
  top: 1.2rem;
  right: 1.5rem;
  font-size: 2.1rem;
  color: #7b68ee;
  cursor: pointer;
  transition: color 0.2s, transform 0.2s;
  z-index: 2;
}

.detail-modal-title {
  color: #34495e;
  font-size: 1.45rem;
  font-weight: 700;
  margin-bottom: 1.7rem;
  letter-spacing: 1px;
  text-align: center;
  background: linear-gradient(90deg, #7b68ee 0%, #6a5acd 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.detail-gene-details {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 1.1rem;
}

.detail-detail-row {
  display: flex;
  align-items: center;
  gap: 1.2rem;
  padding: 0.7rem 0.5rem;
  border-radius: 10px;
  background: linear-gradient(90deg, #fafdff 80%, #e3f0ff 100%);
  box-shadow: 0 1px 4px rgba(123,104,238,0.07);
}

.detail-detail-label {
  min-width: 80px;
  color: #7b68ee;
  font-weight: 600;
  font-size: 1.08rem;
  letter-spacing: 0.5px;
}

.detail-detail-value {
  color: #34495e;
  font-size: 1.08rem;
  font-weight: 500;
  word-break: break-all;
}

</style>