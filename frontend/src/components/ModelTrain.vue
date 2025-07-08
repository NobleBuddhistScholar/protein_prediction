<template>
  <div class="container gradient-bg">
    <div class="header-section">
      <h2 class="section-title">
        <i class="fa-solid fa-brain"></i> 模型训练
      </h2>
      <button class="refresh-btn" @click="refreshPage" :disabled="isUploading">
        <i class="fa-solid fa-sync" :class="{ 'fa-spin': isUploading }"></i> 刷新
      </button>
    </div>

    <!-- 选择模型和上传数据集（一行显示） -->
    <div v-if="!isTraining && !trainingFinished" class="form-row-flex card">
      <div class="form-col">
        <label class="form-label">选择已有模型（可选）：</label>
        <select v-model="selectedModel" class="form-select">
          <option v-for="model in modelList" :key="model" :value="model">{{ model }}</option>
        </select>
      </div>
      
      <div class="form-divider"></div>
      
      <div class="form-col">
        <label class="form-label">上传数据集：</label>
        <div class="file-upload-container">
          <label for="file-upload" class="file-upload-btn">
            <i class="fa-solid fa-cloud-upload-alt"></i> 选择文件
          </label>
          <input id="file-upload" type="file" @change="handleFileChange" class="form-file" accept=".fasta,.fa" webkitdirectory multiple />
          <span v-if="fileName" class="file-name">{{ fileName }}</span>
        </div>
      </div>
    </div>

    <!-- 训练参数输入区 -->
    <div v-if="!isTraining && !trainingFinished" class="form-row card" style="flex-wrap: wrap; gap: 1.5rem;">
      <div>
        <label>max_length：</label>
        <input v-model.number="form.max_length" type="number" min="100" max="100000" step="100" class="input-beauty" style="width:90px" />
      </div>
      <div>
        <label>batch_size：</label>
        <input v-model.number="form.batch_size" type="number" min="1" max="1024" step="1" class="input-beauty" style="width:70px" />
      </div>
      <div>
        <label>epochs：</label>
        <input v-model.number="form.epochs" type="number" min="1" max="1000" step="1" class="input-beauty" style="width:60px" />
      </div>
      <div>
        <label>learning_rate：</label>
        <input v-model.number="form.learning_rate" type="number" min="0.00001" max="1" step="0.00001" class="input-beauty" style="width:90px" />
      </div>
      <div>
        <label>val_ratio：</label>
        <input v-model.number="form.val_ratio" type="number" min="0.01" max="0.99" step="0.01" class="input-beauty" style="width:60px" />
      </div>
    </div>
    <!-- 模型名称输入单独一行 -->
    <div v-if="!isTraining && !trainingFinished" class="model-name-row card">
      <label class="form-label">新模型名：</label>
      <input v-model="form.new_model_name" class="input-beauty model-name-input" placeholder="仅字母数字下划线" maxlength="32" />
    </div>

    <!-- 启动训练按钮 -->
    <div v-if="!isTraining && !trainingFinished" class="form-row card">
      <button class="train-button gradient-btn" :disabled="!fileName || !form.new_model_name || isTraining" @click="startTraining">
        <i class="fa-solid fa-play"></i> 开始训练
      </button>
    </div>

    <!-- 训练进度 -->
    <div v-if="isTraining" class="progress-container card progress-card-narrow">
      <div class="progress-label">训练进度：<span class="progress-percent">{{ progress }}%</span></div>
      <div class="progress-bar-container">
        <div class="progress-bar gradient-progress" :style="{ width: progress + '%' }"></div>
      </div>
    </div>

    <!-- 训练日志/曲线 -->
    <div v-if="epochLogs.length" class="card log-card-narrow">
      <h3 style="margin-bottom:1rem;">训练过程</h3>
      <table v-if="epochLogs.length" class="train-log-table">
        <thead>
          <tr>
            <th>Epoch</th>
            <th>Train Loss</th>
            <th>Train Acc</th>
            <th>Val Loss</th>
            <th>Val Acc</th>
            <th>LR</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="log in epochLogs" :key="log.epoch">
            <td>{{ log.epoch }}</td>
            <td>{{ log.train.loss.toFixed(4) }}</td>
            <td>{{ (log.train.accuracy*100).toFixed(2) }}%</td>
            <td>{{ log.val.loss.toFixed(4) }}</td>
            <td>{{ (log.val.accuracy*100).toFixed(2) }}%</td>
            <td>{{ log.learning_rate.toExponential(2) }}</td>
          </tr>
        </tbody>
      </table>
      <!-- 训练结果展示 -->
      <div v-if="trainingFinished" class="result-section card result-section-beauty">
        <div class="saved-tip saved-tip-beauty">
          <i class="fa-solid fa-check-circle"></i>
          <span>模型已自动保存为 <b>{{ form.new_model_name }}.pth</b> / <b>{{ form.new_model_name }}.json</b></span>
        </div>
      </div>
    </div>

    <div v-if="errorMsg" class="error-tip card"><i class="fa-solid fa-exclamation-triangle"></i> {{ errorMsg }}</div>
  </div>
</template>

<script>
import { API_BASE_URL } from '../config.js';
export default {
  data() {
    return {
      modelList: [
        'HybridModel_v1',
        'HyperFusionCortex_v1',
        'MSA_ResGRUNet_protein_classifier',
        'MSA_ResGRUNet_v1',
        'protein_classifier'
      ],
      selectedModel: 'HybridModel_v1', // 默认选中第一个模型
      fileName: '',
      isTraining: false,
      progress: 0,
      trainingFinished: false,
      metrics: {
        accuracy: '--',
        loss: '--'
      },
      fileList: [],
      errorMsg: '',
      modelBin: null,
      modelJson: null,
      epochLogs: [],
      form: {
        max_length: 10000,
        batch_size: 32,
        epochs: 10,
        learning_rate: 0.0001,
        val_ratio: 0.2,
        new_model_name: ''
      }
    };
  },
  methods: {
    // 刷新整个页面的方法
    refreshPage() {
      this.isUploading = true;
      
      // 显示加载状态
      setTimeout(() => {
        // 重置所有状态数据
        this.resetAll();
        
        // 获取最新的模型列表
        this.modelList = [
          'HybridModel_v1',
          'HyperFusionCortex_v1',
          'MSA_ResGRUNet_protein_classifier',
          'MSA_ResGRUNet_v1',
          'protein_classifier'
        ];
        
        // 恢复状态
        this.isUploading = false;
      }, 500);
    },
    handleFileChange(e) {
      const files = Array.from(e.target.files);
      this.fileList = files;
      this.fileName = files.length > 0 ? `已选择${files.length}个文件` : '';
    },
    async startTraining() {
      this.isTraining = true;
      this.progress = 0;
      this.trainingFinished = false;
      this.errorMsg = '';
      this.metrics = { accuracy: '--', loss: '--' };
      this.epochLogs = [];
      // 构造FormData
      const formData = new FormData();
      if (this.fileList && this.fileList.length > 0) {
        this.fileList.forEach(f => formData.append('file', f));
      }
      if (this.selectedModel) {
        formData.append('pretrain_model', this.selectedModel);
      }
      formData.append('model_type', this.selectedModel ? this.selectedModel.replace('.pth', '').replace('_v1','').replace('_protein_classifier','') : 'HybridModel');
      formData.append('max_length', this.form.max_length);
      formData.append('batch_size', this.form.batch_size);
      formData.append('epochs', this.form.epochs);
      formData.append('learning_rate', this.form.learning_rate);
      formData.append('val_ratio', this.form.val_ratio);
      formData.append('new_model_name', this.form.new_model_name);
      try {
        const url = `${API_BASE_URL}/train_model`;
        const res = await fetch(url, {
          method: 'POST',
          body: formData
        });
        if (!res.body) throw new Error('No response body');
        const reader = res.body.getReader();
        let buffer = '';
        let done = false;
        let totalEpochs = this.form.epochs;
        while (!done) {
          const { value, done: doneReading } = await reader.read();
          done = doneReading;
          if (value) {
            buffer += new TextDecoder().decode(value);
            let lines = buffer.split('\n');
            buffer = lines.pop(); // 最后一行可能不完整
            for (let line of lines) {
              if (!line.trim()) continue;
              let data;
              try { data = JSON.parse(line); } catch { continue; }
              if (data.type === 'epoch') {
                this.epochLogs.push(data);
                this.progress = Math.round((data.epoch / totalEpochs) * 100);
              } else if (data.type === 'done') {
                this.modelBin = data.model_bytes;
                this.modelJson = data.model_json;
                this.trainingFinished = true;
                if (data.model_json && data.model_json.metrics) {
                  this.metrics = {
                    accuracy: data.model_json.metrics.accuracy || '--',
                    loss: data.model_json.metrics.loss || '--'
                  };
                }
                this.progress = 100;
                this.isTraining = false;
              } else if (data.error) {
                this.errorMsg = data.error;
                this.isTraining = false;
                return;
              }
            }
          }
        }
      } catch (err) {
        this.errorMsg = '训练失败：' + err.message;
        this.isTraining = false;
      }
      this.isTraining = false;
    },
    downloadModel() {
      if (!this.modelBin || !this.modelJson) return;
      const bin = new Uint8Array(this.modelBin.match(/.{1,2}/g).map(byte => parseInt(byte, 16)));
      const blob = new Blob([bin], { type: 'application/octet-stream' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = (this.form.new_model_name || 'trained_model') + '.pth';
      a.click();
      // 下载json
      const jsonBlob = new Blob([JSON.stringify(this.modelJson, null, 2)], { type: 'application/json' });
      const a2 = document.createElement('a');
      a2.href = URL.createObjectURL(jsonBlob);
      a2.download = (this.form.new_model_name || 'trained_model') + '.json';
      a2.click();
    },
    resetAll() {
      this.selectedModel = '';
      this.fileName = '';
      this.isTraining = false;
      this.progress = 0;
      this.trainingFinished = false;
      this.form = {
        max_length: 10000,
        batch_size: 32,
        epochs: 10,
        learning_rate: 0.0001,
        val_ratio: 0.2,
        new_model_name: ''
      };
      this.errorMsg = '';
      this.modelBin = null;
      this.modelJson = null;
      this.epochLogs = [];
      this.fileList = [];
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
.gradient-bg {
  background: linear-gradient(120deg, #f5f7fa 0%, #e3f0ff 100%);
  min-height: 100vh;
}
.card {
  background: #fff;
  border-radius: 18px;
  box-shadow: 0 8px 32px rgba(80,120,200,0.10);
  margin-bottom: 2.2rem;
  padding: 2.2rem 2rem 2rem 2rem;
  border: none;
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
.form-row {
  display: flex;
  align-items: center;
  margin-bottom: 1.5rem;
  gap: 1.2rem;
  background: #f8fafd;
  border-radius: 12px;
  padding: 1.1rem 1.2rem;
  box-shadow: 0 2px 8px rgba(80,120,200,0.04);
}
.form-row-flex {
  display: flex;
  align-items: flex-start;
  margin-bottom: 1.5rem;
  gap: 1.5rem;
  background: #f8fafd;
  border-radius: 12px;
  padding: 1.5rem 2rem;
  box-shadow: 0 2px 8px rgba(80,120,200,0.04);
}
.form-col {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
}
.form-divider {
  width: 1px;
  align-self: stretch;
  background: linear-gradient(to bottom, transparent, rgba(80,120,200,0.12), transparent);
  margin: 0 0.5rem;
}
.file-upload-container {
  display: flex;
  align-items: center;
  gap: 1rem;
  flex-wrap: wrap;
}
.file-upload-btn {
  background: linear-gradient(135deg, #4a89dc, #5b9dff);
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 500;
  padding: 0.7rem 1.2rem;
  cursor: pointer;
  transition: all 0.3s;
  box-shadow: 0 2px 8px rgba(74,137,220,0.2);
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
}
.file-upload-btn:hover {
  background: linear-gradient(135deg, #3a79cc, #4a8def);
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(74,137,220,0.3);
}
.form-file {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}
.form-label {
  color: #34495e;
  font-weight: 600;
  font-size: 1.08rem;
  margin-bottom: 0.3rem;
  display: block;
  letter-spacing: 0.3px;
}
.form-select {
  appearance: none;
  -webkit-appearance: none;
  background: linear-gradient(90deg, #f5f7fa 80%, #e3f0ff 100%);
  border: 1.5px solid #b0c4de;
  color: #34495e;
  font-size: 1.08rem;
  font-weight: 500;
  padding: 0.7rem 2.5rem 0.7rem 1rem;
  border-radius: 10px;
  box-shadow: 0 2px 8px rgba(80,120,200,0.06);
  transition: all 0.3s;
  outline: none;
  background-image: url('data:image/svg+xml;utf8,<svg fill="%234a89dc" height="18" viewBox="0 0 24 24" width="18" xmlns="http://www.w3.org/2000/svg"><path d="M7 10l5 5 5-5z"/></svg>');
  background-repeat: no-repeat;
  background-position: right 0.8rem center;
  background-size: 1.2rem;
  min-width: 220px;
  max-width: 100%;
  cursor: pointer;
}
.form-select:focus, .form-select:hover {
  border-color: #4a89dc;
  box-shadow: 0 0 0 2px rgba(74,137,220,0.2);
  transform: translateY(-1px);
}
.form-row input[type=number], .form-row input[type=text] {
  background: linear-gradient(90deg, #f5f7fa 80%, #e3f0ff 100%);
  border: 1.5px solid #b0c4de;
  color: #34495e;
  font-size: 1.08rem;
  font-weight: 500;
  padding: 0.6rem 1rem;
  border-radius: 10px;
  box-shadow: 0 2px 8px rgba(80,120,200,0.06);
  transition: border 0.3s, box-shadow 0.3s;
  outline: none;
}
.form-row input[type=number]:focus, .form-row input[type=text]:focus {
  border-color: #4a89dc;
  box-shadow: 0 0 0 2px #b0c4de33;
}
.gradient-btn {
  background: linear-gradient(135deg, #4a89dc, #6dd5ed);
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 1.13rem;
  font-weight: 600;
  padding: 0.8rem 2.1rem;
  cursor: pointer;
  transition: background 0.3s;
  box-shadow: 0 2px 8px rgba(80,120,200,0.08);
}
.gradient-btn:disabled {
  background: #b0c4de;
  cursor: not-allowed;
}
.train-button i, .save-btn i, .download-btn i {
  margin-right: 0.5rem;
}
.progress-container {
  width: 600;
  height: 22px;
  background-color: #e0e0e0;
  border-radius: 12px;
  margin: 1.5rem 0;
  position: relative;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(80,120,200,0.04);
}
.progress-card-narrow {
  max-width: 100%;
  margin: 1.5rem auto;
  padding: 1.1rem 0.7rem 1rem 0.7rem;
  border-radius: 14px;
}
.gradient-progress {
  position: absolute;
  left: 0;
  top: 0;
  height: 100%;
  background: linear-gradient(90deg, #4a89dc, #6dd5ed);
  transition: width 0.3s;
}
.progress-text {
  position: absolute;
  right: 10px;
  top: -22px;
  font-size: 1rem;
  color: #4a89dc;
  font-weight: 600;
}
.train-log-table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  margin-bottom: 1rem;
  background: #fafdff;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(80,120,200,0.04);
}
.result-section-beauty {
  background: linear-gradient(135deg, #c175e1 0%, #731d80);
  border-radius: 18px;
  box-shadow: 0 8px 32px rgba(80,120,200,0.10);
  padding: 2.2rem 2rem 2rem 2rem;
  margin-top: 2rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  width: 700px;
  height: 20px;
  margin-left: auto;
  margin-right: auto;
}
.saved-tip-beauty {
  margin-top: 0;
  background: transparent;
  border: 1.5px solid #b2dfdb00;
  border-radius: 14px;
  box-shadow: 0 2px 8px rgba(67,160,71,0.07);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 1.2rem;
  padding: 1.1rem 1.5rem;
  font-size: 1.13rem;
  color: #fff;
}
.saved-tip-beauty i.fa-check-circle {
  color: #fff;
  font-size: 1.6rem;
}
.saved-tip-beauty span {
  color: #fff;
  font-weight: 500;
}
.cancel-btn {
  padding: 0.7rem 1.5rem;
  background: linear-gradient(135deg, #f44336, #e57373);
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 500;
  transition: background 0.3s;
  box-shadow: 0 2px 8px rgba(244,67,54,0.08);
}
.cancel-btn:hover {
  background: #e53935;
}
.file-name {
  color: #4a89dc;
  font-size: 0.98rem;
  padding: 0.4rem 1rem;
  background: rgba(74,137,220,0.1);
  border-radius: 6px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 200px;
  display: inline-block;
}
.error-tip {
  color: #e53935;
  font-size: 1rem;
  margin-top: 1rem;
  text-align: center;
  background: #fff3f3;
  border-radius: 8px;
  padding: 1rem;
  border: 1px solid #ffcdd2;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  justify-content: center;
}
.input-beauty {
  background: linear-gradient(90deg, #f5f7fa 80%, #e3f0ff 100%);
  border: 1.5px solid #b0c4de;
  color: #34495e;
  font-size: 1.08rem;
  font-weight: 500;
  padding: 0.6rem 1rem;
  border-radius: 10px;
  box-shadow: 0 2px 8px rgba(80,120,200,0.06);
  transition: border 0.3s, box-shadow 0.3s;
  outline: none;
}
.input-beauty:focus {
  border-color: #4a89dc;
  box-shadow: 0 0 0 2px #b0c4de33;
}
.model-name-row {
  display: flex;
  align-items: center;
  gap: 1.2rem;
  background: #f8fafd;
  border-radius: 12px;
  padding: 1.1rem 1.2rem;
  box-shadow: 0 2px 8px rgba(80,120,200,0.04);
  margin-bottom: 1.5rem;
}
.model-name-input {
  min-width: 220px;
  max-width: 320px;
  flex: 1;
}
/* 训练日志区专用样式 */
.log-card-narrow {
  max-width: 800px;
  margin: 0 auto 1.8rem auto;
  padding: 2rem 2rem 1.8rem 2rem;
  border-radius: 14px;
  overflow-x: auto; /* 添加水平滚动条 */
  background: linear-gradient(145deg, #ffffff, #f8faff);
  box-shadow: 0 10px 30px rgba(74, 137, 220, 0.1);
  border: 1px solid rgba(74, 137, 220, 0.06);
}

.log-card-narrow h3 {
  color: #2c3e50;
  font-size: 1.4rem;
  margin-bottom: 1.4rem;
  text-align: center;
  font-weight: 600;
  letter-spacing: 0.6px;
  position: relative;
  display: inline-block;
  padding-bottom: 0.5rem;
  left: 50%;
  transform: translateX(-50%);
}

.log-card-narrow h3::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 25%;
  width: 50%;
  height: 2px;
  background: linear-gradient(90deg, transparent, #4a89dc, transparent);
}

/* 强化表格宽度控制 */
.log-card-narrow .train-log-table {
  table-layout: fixed;
  width: 100%;
  max-width: 820px;
  box-shadow: 0 4px 16px rgba(80,120,200,0.1);
  border-radius: 12px;
  border: 1px solid rgba(80,120,200,0.07);
  margin: 0 auto;
}

.log-card-narrow .train-log-table th,
.log-card-narrow .train-log-table td {
  padding: 0.85rem 0.6rem;
  font-size: 1rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  text-align: center;
  transition: all 0.3s ease;
  border-bottom: 1px solid rgba(80,120,200,0.05);
}

.log-card-narrow .train-log-table th {
  background: linear-gradient(90deg, #4a89dc 0%, #5b9dff 100%);
  color: #ffffff;
  font-weight: 600;
  padding: 1rem 0.6rem;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  font-size: 0.95rem;
  border-bottom: none;
}

.log-card-narrow .train-log-table tr:hover td {
  background-color: #f0f7ff;
  transform: translateY(-1px);
  box-shadow: 0 2px 5px rgba(74, 137, 220, 0.1);
}

/* 设置各列宽度 */
.log-card-narrow .train-log-table th:nth-child(1),
.log-card-narrow .train-log-table td:nth-child(1) {
  width: 60px; /* Epoch列宽度 */
}

.log-card-narrow .train-log-table th:nth-child(2),
.log-card-narrow .train-log-table td:nth-child(2) {
  width: auto; /* Train Loss列宽度 */
}

.log-card-narrow .train-log-table th:nth-child(3),
.log-card-narrow .train-log-table td:nth-child(3) {
  width: auto; /* Val Loss列宽度 */
}

.log-card-narrow .train-log-table th:nth-child(4),
.log-card-narrow .train-log-table td:nth-child(4) {
  width: auto; /* Train Acc列宽度 */
}

.log-card-narrow .train-log-table th:nth-child(5),
.log-card-narrow .train-log-table td:nth-child(5) {
  width: auto; /* Val Acc列宽度 */
}

.log-card-narrow .train-log-table th:nth-child(6),
.log-card-narrow .train-log-table td:nth-child(6) {
  width: 80px; /* LR列宽度 */
}

.log-card-narrow .train-log-table tr:nth-child(even) td {
  background-color: #f5f9ff;
}

.log-card-narrow .train-log-table tr:last-child td:first-child {
  border-bottom-left-radius: 12px;
}

.log-card-narrow .train-log-table tr:last-child td:last-child {
  border-bottom-right-radius: 12px;
}

.log-card-narrow .train-log-table tr:first-child th:first-child {
  border-top-left-radius: 12px;
}

.log-card-narrow .train-log-table tr:first-child th:last-child {
  border-top-right-radius: 12px;
}

/* 进度条容器专用样式 */
.progress-card-narrow {
  max-width: 750px;
  margin: 1.5rem auto;
  padding: 1.8rem 2rem 1.5rem 2rem;
  border-radius: 14px;
  background: linear-gradient(145deg, #ffffff, #f8faff);
  box-shadow: 0 10px 30px rgba(74, 137, 220, 0.12);
  border: 1px solid rgba(74, 137, 220, 0.08);
}

.progress-card-narrow .progress-label {
  display: flex;
  justify-content: space-between;
  margin-bottom: 0.7rem;
  color: #34495e;
  font-size: 1.1rem;
  font-weight: 400;
  letter-spacing: 0.5px;
}

.progress-card-narrow .progress-percent {
  background: linear-gradient(90deg, #4a89dc 0%, #6dd5ed 100%);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  font-weight: 700;
  font-size: 1.05rem;
}

.progress-card-narrow .progress-bar-container {
  width: 100%;
  height: 8px;
  background-color: #e8eef7;
  border-radius: 10px;
  position: relative;
  overflow: hidden;
  box-shadow: inset 0 1px 4px rgba(0,0,0,0.08);
  margin-bottom: 0.8rem;
}

.progress-card-narrow .progress-bar {
  position: absolute;
  left: 0;
  top: 0;
  height: 100%;
  border-radius: 10px;
  background: linear-gradient(90deg, #4a89dc 20%, #5b9dff 70%, #6dd5ed 100%);
  box-shadow: 0 1px 8px rgba(74,137,220,0.4);
  transition: width 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0% {
    box-shadow: 0 0 0 0 rgba(74, 137, 220, 0.4);
  }
  70% {
    box-shadow: 0 0 0 6px rgba(74, 137, 220, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(74, 137, 220, 0);
  }
}

@media (max-width: 900px) {
  .container {
    padding: 1rem;
  }
  .card {
    padding: 1.1rem 0.5rem;
  }
  .result-section-beauty {
    padding: 1.2rem 0.5rem;
  }
  .saved-tip-beauty {
    flex-direction: column;
    gap: 0.7rem;
    font-size: 1rem;
  }
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
</style>