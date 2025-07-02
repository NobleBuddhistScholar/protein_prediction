<template>
  <div class="container gradient-bg">
    <div class="header-section card">
      <h2 class="section-title">
        <i class="fa-solid fa-brain"></i> 预训练模型训练
      </h2>
    </div>

    <!-- 选择模型 -->
    <div class="form-row card">
      <label class="form-label">选择模型：</label>
      <select v-model="selectedModel" class="form-select">
        <option disabled value="">请选择模型</option>
        <option v-for="model in modelList" :key="model" :value="model">{{ model }}</option>
      </select>
    </div>

    <!-- 上传数据集 -->
    <div class="form-row card">
      <label class="form-label">上传数据集：</label>
      <input type="file" @change="handleFileChange" class="form-file" accept=".fasta,.fa" webkitdirectory multiple />
      <span v-if="fileName" class="file-name">{{ fileName }}</span>
    </div>

    <!-- 启动训练按钮 -->
    <div class="form-row card">
      <button class="train-button gradient-btn" :disabled="!selectedModel || !fileName" @click="startTraining">
        <i class="fa-solid fa-play"></i> 开始训练
      </button>
    </div>

    <!-- 训练进度 -->
    <div v-if="isTraining" class="progress-container card">
      <div class="progress-bar gradient-progress" :style="{ width: progress + '%' }"></div>
      <span class="progress-text">{{ progress }}%</span>
    </div>

    <!-- 训练结果展示 -->
    <div v-if="trainingFinished" class="result-section card">
      <h3 class="result-title"><i class="fa-solid fa-chart-line"></i> 训练结果</h3>
      <div class="metrics">
        <div>准确率：{{ metrics.accuracy }}%</div>
        <div>损失：{{ metrics.loss }}</div>
      </div>
      <!-- 训练曲线图（占位） -->
      <div class="chart-placeholder">
        <span>训练曲线图（如loss/accuracy）</span>
      </div>
    </div>

    <div v-if="trainDone && !modelSaved" class="save-section card">
      <h3>训练已完成，是否保存模型？</h3>
      <div class="save-input-row">
        <input v-model="saveModelName" placeholder="请输入新模型名称（仅字母数字下划线）" class="save-input" maxlength="32" />
        <button class="save-btn gradient-btn" :disabled="!saveModelName || isSaving" @click="uploadModel"><i class="fa-solid fa-upload"></i> 保存到模型库</button>
        <button class="cancel-btn" @click="resetAll">放弃</button>
      </div>
      <div v-if="saveError" class="error-tip">{{ saveError }}</div>
    </div>
    <div v-if="modelSaved" class="saved-tip card">
      <span><i class="fa-solid fa-check-circle"></i> 模型和配置已保存！</span>
      <button class="download-btn gradient-btn" @click="downloadModel"><i class="fa-solid fa-download"></i> 下载到本地</button>
      <button class="cancel-btn" @click="resetAll">关闭</button>
    </div>
    <div v-if="errorMsg" class="error-tip card"><i class="fa-solid fa-exclamation-triangle"></i> {{ errorMsg }}</div>
  </div>
</template>

<script>
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
      selectedModel: '',
      fileName: '',
      isTraining: false,
      progress: 0,
      trainingFinished: false,
      metrics: {
        accuracy: 98.5,
        loss: 0.023
      },
      saveModelName: '',
      isSaving: false,
      saveError: '',
      savedPth: '',
      savedJson: '',
      fileList: []
    };
  },
  methods: {
    handleFileChange(e) {
      const files = Array.from(e.target.files);
      this.fileList = files;
      this.fileObj = files.length > 0 ? files : null;
      this.fileName = files.length > 0 ? `已选择${files.length}个文件` : '';
    },
    startTraining() {
      this.isTraining = true;
      this.progress = 0;
      this.evalRows = [];
      this.trainDone = false;
      this.modelSaved = false;
      this.modelBin = null;
      this.modelJson = null;
      this.errorMsg = '';
      // 构造FormData
      const formData = new FormData();
      if (this.fileList && this.fileList.length > 0) {
        this.fileList.forEach(f => formData.append('file', f));
      }
      formData.append('model_type', this.selectedModel.replace('.pth', '').replace('_v1','').replace('_protein_classifier',''));
      if (this.hyperparams) {
        Object.entries(this.hyperparams).forEach(([k, v]) => formData.append(k, v));
      }
      const url = 'http://localhost:5000/train_model';
      fetch(url, {
        method: 'POST',
        body: formData
      }).then(async res => {
        const data = await res.json();
        if (data.error) {
          this.errorMsg = data.error;
          this.isTraining = false;
          return;
        }
        this.trainDone = true;
        this.isTraining = false;
        this.modelBin = data.model_bytes;
        this.modelJson = data.model_json;
        this.progress = 100;
        this.trainingFinished = true;
        // 可选：展示训练指标
        if (data.model_json && data.model_json.metrics) {
          this.metrics = {
            accuracy: data.model_json.metrics.accuracy || '--',
            loss: data.model_json.metrics.loss || '--'
          };
        }
      }).catch(err => {
        this.errorMsg = '训练失败：' + err.message;
        this.isTraining = false;
      });
    },
    uploadModel() {
      this.isSaving = true;
      this.saveError = '';
      // 构造FormData
      const formData = new FormData();
      // 模型二进制转Blob
      const bin = new Uint8Array(this.modelBin.match(/.{1,2}/g).map(byte => parseInt(byte, 16)));
      const blob = new Blob([bin], { type: 'application/octet-stream' });
      formData.append('model_file', blob, this.saveModelName + '.pth');
      formData.append('config_json', JSON.stringify(this.modelJson, null, 2));
      formData.append('model_name', this.saveModelName);
      fetch('http://localhost:5000/save_model', {
        method: 'POST',
        body: formData
      }).then(async res => {
        const data = await res.json();
        if (!res.ok) {
          this.saveError = data.error || '保存失败';
          this.isSaving = false;
          return;
        }
        this.modelSaved = true;
        this.savedPth = data.pth;
        this.savedJson = data.json;
        this.isSaving = false;
      }).catch(err => {
        this.saveError = '保存失败：' + err.message;
        this.isSaving = false;
      });
    },
    downloadModel() {
      // 下载pth
      if (!this.modelBin || !this.modelJson) return;
      const bin = new Uint8Array(this.modelBin.match(/.{1,2}/g).map(byte => parseInt(byte, 16)));
      const blob = new Blob([bin], { type: 'application/octet-stream' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = (this.saveModelName || this.modelJson.display_name || 'trained_model') + '.pth';
      a.click();
      // 下载json
      const jsonBlob = new Blob([JSON.stringify(this.modelJson, null, 2)], { type: 'application/json' });
      const a2 = document.createElement('a');
      a2.href = URL.createObjectURL(jsonBlob);
      a2.download = (this.saveModelName || this.modelJson.display_name || 'trained_model') + '.json';
      a2.click();
    },
    resetAll() {
      this.selectedModel = '';
      this.fileName = '';
      this.isTraining = false;
      this.progress = 0;
      this.trainingFinished = false;
      this.saveModelName = '';
      this.isSaving = false;
      this.saveError = '';
      this.modelSaved = false;
      this.savedPth = '';
      this.savedJson = '';
      this.errorMsg = '';
      this.trainDone = false; // 重置训练完成标志
    }
  }
};
</script>

<style scoped>
.container {
  max-width: 700px;
  margin: 0 auto;
  padding: 2rem;
  font-family: 'Arial', sans-serif;
}
.gradient-bg {
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
.section-title {
  color: #2c3e50;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin: 0;
  font-size: 1.8rem;
  font-weight: 600;
}
.form-row {
  display: flex;
  align-items: center;
  margin-bottom: 1.5rem;
  gap: 1rem;
}
.form-label {
  min-width: 100px;
  color: #34495e;
  font-weight: 500;
}
.form-select, .form-file {
  padding: 0.5rem 1rem;
  border-radius: 8px;
  border: 1.5px solid #e0e0e0;
  font-size: 1rem;
  background: #f8f9fa;
  transition: border 0.3s;
}
.form-select:focus, .form-file:focus {
  border-color: #4a89dc;
  outline: none;
}
.gradient-btn {
  background: linear-gradient(135deg, #4a89dc, #6dd5ed);
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 1.1rem;
  font-weight: 600;
  padding: 0.8rem 2rem;
  cursor: pointer;
  transition: background 0.3s;
}
.gradient-btn:disabled {
  background: #b0c4de;
  cursor: not-allowed;
}
.train-button i, .save-btn i, .download-btn i {
  margin-right: 0.5rem;
}
.progress-container {
  width: 100%;
  height: 20px;
  background-color: #e0e0e0;
  border-radius: 10px;
  margin: 1.5rem 0;
  position: relative;
  overflow: hidden;
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
  top: -28px;
  font-size: 1rem;
  color: #7f8c8d;
}
.result-section {
  margin-top: 2rem;
  background: #f9f9f9;
  border-radius: 12px;
  padding: 1.5rem;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.result-title {
  color: #2c3e50;
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.metrics {
  font-size: 1.1rem;
  margin-bottom: 1.5rem;
  color: #34495e;
}
.chart-placeholder {
  height: 220px;
  background: #e0e0e0;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #7f8c8d;
  font-size: 1.2rem;
}
.file-name {
  color: #4a89dc;
  font-size: 0.95rem;
  margin-left: 1rem;
}
.save-section {
  margin-top: 1.5rem;
  padding: 1.5rem;
  background: #f1f8e9;
  border-radius: 12px;
  border: 1px solid #c8e6c9;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.save-input-row {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-top: 1.2rem;
  justify-content: center;
}
.save-input {
  padding: 0.7rem 1.2rem;
  border-radius: 8px;
  border: 1.5px solid #b0c4de;
  font-size: 1.05rem;
  min-width: 220px;
}
.save-btn {
  padding: 0.7rem 1.5rem;
  background: linear-gradient(135deg, #4caf50, #43e97b);
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 500;
  transition: background 0.3s;
}
.save-btn:disabled {
  background: #a5d6a7;
  cursor: not-allowed;
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
}
.cancel-btn:hover {
  background: #e53935;
}
.saved-tip {
  margin-top: 1.5rem;
  padding: 1rem;
  background: #e8f5e9;
  border-radius: 12px;
  border: 1px solid #c8e6c9;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
}
.download-btn {
  padding: 0.8rem 2rem;
  background: linear-gradient(135deg, #4a89dc, #6dd5ed);
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 1.1rem;
  font-weight: 600;
  margin-right: 1.5rem;
  cursor: pointer;
  transition: background 0.3s;
}
.download-btn:hover {
  background: linear-gradient(135deg, #357ab8, #4a89dc);
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
@media (max-width: 768px) {
  .container {
    padding: 1rem;
  }
  .card {
    padding: 1rem 0.5rem;
  }
  .result-section, .save-section, .saved-tip {
    padding: 1rem;
  }
  .save-input {
    min-width: 120px;
  }
}
</style>