<template>
  <div class="container">
    <!-- 头部部分 -->
    <div class="header-section">
      <h2 class="section-title">
        <i class="fa-solid fa-microscope"></i> 基因组注释
      </h2>
      <div class="model-select-container">
        <label class="model-select-label">选择模型：</label>
        <select v-model="selectedModelFile" class="model-select">
          <option v-for="option in modelOptions" :key="option.value" :value="option.value">
            {{ option.label }}
          </option>
        </select>
      </div> 
      <div class="file-upload-container">
        <label class="file-upload-button">
          <input type="file" ref="fileInput" @change="handleFileUpload" accept=".fasta" />
          <span v-if="!isUploading &&!uploadError">
            <i class="icon-upload"></i> {{ fileName || '选择FASTA文件' }}
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
          <button @click="showReportModal">
            <i class="fa-solid fa-table-list"></i> 查看txt报告
          </button>
          <button @click="openGffReportModal"> <!-- 修改方法名 -->
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

        <!-- 报告模态框 -->
        <div v-if="showModal" class="modal">
          <div class="modal-content">
            <span class="close" @click="closeModal">&times;</span>
            <h3 class="modal-title">检测报告</h3>
            <!-- 修改这里，调用 formatContent 方法并传入 currentReport -->
            <div v-if="currentReport" class="report-content">
              <div v-html="formatContent(currentReport)"></div>
            </div>
            <div v-else>正在加载报告...</div>
          </div>
        </div>
        <!-- 基因组详细信息模态框 -->
        <div v-if="showGeneDetailsModal" class="modal">
          <div class="modal-content">
            <span class="close" @click="closeGeneDetailsModal">&times;</span>
            <h3 class="modal-title">基因组详细信息</h3>
            <div class="gene-details">
              <div class="detail-row">
                <span class="detail-label">类型:</span>
                <span class="detail-value">{{ currentGeneDetails.type }}</span>
              </div>
              <div class="detail-row">
                <span class="detail-label">位置:</span>
                <span class="detail-value">{{ currentGeneDetails.location }}</span>
              </div>
              <div class="detail-row">
                <span class="detail-label">链:</span>
                <span class="detail-value">{{ currentGeneDetails.strand }}</span>
              </div>
              <div class="detail-row">
                <span class="detail-label">置信度:</span>
                <span class="detail-value">{{ currentGeneDetails.qualifiers.confidence }}</span>
              </div>
              <div class="detail-row">
                <span class="detail-label">长度:</span>
                <span class="detail-value">{{ currentGeneDetails.qualifiers.protein_length }}</span>
              </div>
            </div>
          </div>
        </div>
        <!-- GFF报告模态框 -->
        <div v-if="showGffReportModal" class="modal">
          <div class="modal-content gff-modal-content">
            <span class="close" @click="closeGffReportModal">&times;</span>
          <!-- 新增一个容器来包含标题和下载按钮 -->
          <div class="gff-title-button-container">
              <h2 class="modal-title">GFF报告</h2>
              <!-- 修改：添加新的类名 -->
              <button @click="downloadGffReport" v-if="gffReport" class="download-gff-button">
                <i class="fa-solid fa-download"></i> 下载GFF报告
              </button>
            </div>
            <div v-if="gffReport" class="report-content">
              <!-- 结构化GFF内容 -->
              <template v-if="typeof gffReport === 'string'">
                <div v-for="(line, index) in gffReport.split('\n')" :key="index">
                  <!-- 处理元数据行 -->
                  <div v-if="line.startsWith('##')" class="gff-header">
                    {{ line.slice(2) }}
                  </div>
                  <!-- 处理数据行 -->
                  <div v-else class="gff-row">
                    <span 
                      v-for="(cell, cellIndex) in line.split('\t')" 
                      :key="cellIndex" 
                      class="gff-cell"
                    >
                      {{ cell }}
                    </span>
                  </div>
                </div>
              </template>
            </div>
            <div v-else>正在加载GFF报告...</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      // 新增状态存储报告信息
      reportInfo: {
        genome_id: '',
        report_date: ''
      },
      geneData: null,
      showModal: false,
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
      modelOptions: [] // 模型选项列表
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
    // 新增：加载模型列表
    async loadModelList() {
      try {
        const response = await fetch('http://localhost:5000/models');
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

    async handleFileUpload(event) {
      const file = event.target.files[0];
      if (!file) return;

      this.fileName = file.name;
      this.isUploading = true;
      this.progress = 0;
      this.uploadError = null;

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
        const response = await fetch('http://localhost:5000/upload', {
          method: 'POST',
          body: formData
        });

        if (!response.ok) throw new Error('上传失败');

        const data = await response.json();
        this.geneData = data.results;
        this.currentReport = data.summary;
        this.processGeneRows();

        // 新增：从响应头获取报告文件名
    // 修改：直接从响应数据中获取文件名（假设字段为 filename）
        // 新增：从响应头获取报告文件名
        let filename = '';
        if (data.summary && data.summary.filename) {
          filename = data.summary.filename;
        } else if (data.summary) {
          filename = Object.keys(data.summary)[0];
        } else {
          filename = this.extractFilename(data);
        }

        // 解析基因组ID和日期
        const [genome_id, report_date] = this.parseFilename(filename);

        this.reportInfo = {
          genome_id,
          report_date
        };

        // 完成进度
        this.progress = 100;
        setTimeout(() => {
          this.isUploading = false;
          clearInterval(this.progressInterval);
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

    //获取报告逻辑
    // 修改后的报告获取方法
    async showReportModal() {
      this.showModal = true;
      if (!this.currentReport) {
        try {
      // 从geneData中获取genome_id
          const genome_id = this.geneData.metadata.genome_id;

      // 生成当前日期
          const current_date = new Date().toISOString().split('T')[0];

          const url = `http://localhost:5000/summary?genome_id=${genome_id}&current_date=${current_date}`;
          const response = await fetch(url);

          if (!response.ok) throw new Error('报告获取失败');

          const reportData = await response.json();
          this.currentReport = reportData[Object.keys(reportData)[0]];
        } catch (error) {
          console.error('报告加载失败:', error);
          this.currentReport = `无法加载报告内容：${error.message}`;
        }
      }
    },

    async openGffReportModal() {
    this.showGffReportModal = true;
    this.isGeneratingGff = true;
    try {
      // 从geneData中获取genome_id
      const genome_id = this.geneData.metadata.genome_id;

      // 生成当前日期
      const current_date = new Date().toISOString().split('T')[0];

      // 发出生成GFF报告的请求
      const generateResponse = await fetch(
        `http://localhost:5000/generate_gff?genome_id=${genome_id}&current_date=${current_date}`
      );
      if (!generateResponse.ok) {
        throw new Error('生成GFF报告失败');
      }

      // 获取GFF报告
      const gffResponse = await fetch(
        `http://localhost:5000/gff?genome_id=${genome_id}&current_date=${current_date}`
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

    closeModal() {
      this.showModal = false;
    },

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
        NSP1: '#FF6B6B',
        NSP2: '#4ECDC4',
        NSP3: '#45B7D1',
        NSP4: '#FFBE0B',
        NSP5: '#FB5607',
        NSP6: '#8338EC',
        NSP7: '#3A86FF',
        NSP8: '#FF006E',
        NSP9: '#88D18A',
        NSP10: '#118AB2',
        NSP11: '#8A89C0',
        NSP12: '#EF476F',
        NSP13: '#06D6A0',
        NSP14: '#1B9AAA',
        NSP15: '#FFC43D',
        NSP16: '#8A89b0',
        membrane_protein: '#E2F0F9',
        envelope_protein: '#FCD5CE',
        nucleocapsid_protein: '#FFF1CC'
      };
      return colors[type] || '#CCCCCC';
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
        const blob = new Blob([this.gffReport], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'gff_report.gff';
        a.click();
        URL.revokeObjectURL(url);
      }
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
.card {
  background: white;
  border-radius: 16px;
  box-shadow: 0 8px 30px rgba(0,0,0,0.08);
  margin-bottom: 2rem;
  padding: 2rem 1.5rem;
}

/* 新增刻度尺样式 */
.ruler-canvas {
  display: block;
  margin-bottom: -10px;
  background: #f8f9fa;
  border-radius: 4px;
}


.gene-length-bar {
  height: 4px;
  background: #e0e0e0;
  border-radius: 2px;
  margin-top: 26px; /* 与canvas高度对齐 */
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
margin-bottom: 1.5rem;
flex-wrap: wrap;
gap: 1rem;
padding: 1rem;
background-color: #f8f9fa;
border-radius: 8px;
box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.section-title {
color: #2c3e50;
display: flex;
align-items: center;
gap: 0.5rem;
margin: 0;
}

.file-upload-container {
display: flex;
flex-direction: column;
gap: 0.5rem;
min-width: 250px;
}

.file-upload-button {
display: inline-flex;
align-items: center;
justify-content: center;
padding: 0.75rem 1.5rem;
background-color: #3498db;
color: white;
border-radius: 6px;
cursor: pointer;
transition: all 0.3s;
font-weight: 500;
text-align: center;
border: none;
position: relative;
overflow: hidden;
}

.file-upload-button:hover {
background-color: #2980b9;
transform: translateY(-1px);
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
height: 6px;
background-color: #e0e0e0;
border-radius: 3px;
position: relative;
overflow: hidden;
}

.progress-bar {
position: absolute;
left: 0;
top: 0;
height: 100%;
background-color: #3d7bdf;
transition: width 0.3s ease;
}

.progress-text {
position: absolute;
right: 0;
top: -20px;
font-size: 0.8rem;
color: #7f8c8d;
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
gap: 2rem;
}

.control-buttons button {
padding: 12px 20px;
background-color: #4a89dc;
color: white;
border: none;
border-radius: 4px;
cursor: pointer;
transition: background-color 0.3s;
}

.control-buttons button:hover {
background-color: #3b7dd8;
}

/* 基因可视化包装器 */
.gene-visualization-wrapper {
border: 1px solid #e0e0e0;
border-radius: 8px;
padding: 15px;
background-color: #f9f9f9;
margin-top: 20px;
overflow-y: hidden;
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
border: 1px solid #e0e0e0;
border-radius: 4px;
background-color: white;
padding: 10px 0;
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
top: 0;
left: 0;
width: 100%;
height: 100%;
background: rgba(0, 0, 0, 0.5);
display: flex;
justify-content: center;
align-items: center;
z-index: 1000;
}

.modal-content {
background: white;
padding: 2rem;
border-radius: 12px;
width: 80%;
max-width: 700px;
max-height: 80vh;
overflow-y: auto;
position: relative;
box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
}

.modal-title {
color: #2c3e50;
margin-bottom: 1.5rem;
padding-bottom: 0.5rem;
border-bottom: 2px solid #f0f0f0;
}

.close {
position: absolute;
right: 1.5rem;
top: 1.5rem;
font-size: 1.8rem;
cursor: pointer;
color: #7f8c8d;
transition: color 0.3s;
}

.close:hover {
color: #2c3e50;
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
background-color: rgba(149, 178, 247, 0.8);
color: white;
padding: 4px 8px;
border-radius: 4px;
font-size: 12px;
z-index: 10;  /* 确保在顶层 */
pointer-events: none;
white-space: nowrap;
}

/*垂直虚线*/
.ruler-line {
  position: absolute;
  width: 3px;
  top: 0px;       /* 下移与基因可视化区域对齐 */
  bottom: -10px;   /* 延伸至基因行区域 */
  background-color: rgba(177, 201, 250, 0.5);
  background-image: linear-gradient(to bottom, rgba(19, 64, 161, 0.5) 50%, transparent 50%);
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
  padding: 12px 20px;
  border: 2px solid #e0e0e0;
  border-radius: 8px;
  font-size: 15px;
  width: 350px;
  background-color: #f8f9fa;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

/* 搜索框聚焦样式 */
input[type="text"]:focus {
  outline: none;
  border-color: #4a89dc;
  box-shadow: 0 0 0 3px rgba(74, 137, 220, 0.1);
}

/* 添加搜索图标 */
input[type="text"] {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' fill='%23999' viewBox='0 0 16 16'%3E%3Cpath d='M11.742 10.344a6.5 6.5 0 1 0-1.397 1.398h-.001c.03.04.062.078.098.115l3.85 3.85a1 1 0 0 0 1.415-1.414l-3.85-3.85a1.007 1.007 0 0 0-.115-.1zM12 6.5a5.5 5.5 0 1 1-11 0 5.5 5.5 0 0 1 11 0z'/%3E%3C/svg%3E");
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
  color: #2c3e50; /* 与网页整体标题颜色一致 */
  font-family: 'Arial', sans-serif; /* 与网页整体字体一致 */
  font-size: 1.5rem; /* 调整字体大小 */
  font-weight: bold; /* 加粗字体 */
  margin-top: 2.5rem; /* 增加顶部间距 */
  margin-bottom: 0.5rem; /* 增加底部间距 */
  padding-top: 1rem; /* 增加底部内边距 */
  border-top: 2px solid #f0f0f0; /* 添加顶边框 */
}


.gene-visualization-title i {
  margin-right: 10px;
  color: #438ddcdf; /* 图标颜色 */
  font-size: 40px; /* 图标大小 */
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
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}
.model-select-label {
  font-weight: 500;
  color: #34495e;
}
.model-select {
  padding: 0.5rem 1rem;
  border-radius: 6px;
  border: 1px solid #e0e0e0;
  font-size: 1rem;
}
</style>