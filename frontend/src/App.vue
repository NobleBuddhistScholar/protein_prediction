<template>
  <div id="app">
    <div class="header">
      <h1>Virgenta</h1>
      <p>基于HyperFusion cortex(HFC)的轻量病毒基因组注释系统-高效解析泛基因组非核心基因突变</p>
      <p>全自动预测病毒基因中的蛋白质序列并生成报告</p>
    </div>
    <div class="main-container">
      <div class="sidebar">
        <button @click="showPage('GeneVisualization')" :class="{ active: currentPage === 'GeneVisualization' }">
          <i class="fa-solid fa-flask-vial"></i> 基因组注释
        </button>
        <button @click="showPage('ModelTrain')" :class="{ active: currentPage === 'ModelTrain' }">
          <i class="fa-solid fa-brain"></i>  模型训练
        </button>
        <button @click="showPage('SummaryReport')" :class="{ active: currentPage === 'SummaryReport' }">
          <i class="fa-solid fa-folder-open"></i>  总报告管理
        </button>
        <button @click="showPage('ModelManage')" :class="{ active: currentPage === 'ModelManage' }">
          <i class="fa-solid fa-folder-open"></i>  模型管理
        </button>
        <button @click="showPage('KnowledgeManage')" :class="{ active: currentPage === 'KnowledgeManage' }">
          <i class="fa-solid fa-folder-open"></i>  知识库管理
        </button>
      </div>
      <div class="content">
        <transition name="fade" mode="out-in">
          <component :is="currentPage"></component>
        </transition>
      </div>
    </div>
  </div>
</template>

<script>
import GeneVisualization from './components/GeneVisualization.vue';
import SummaryReport from './components/SummaryReport.vue';
import ModelTrain from './components/ModelTrain.vue';
import ModelManage from './components/ModelManage.vue';
import KnowledgeManage from './components/KnowledgeManage.vue';

export default {
  data() {
    return {
      currentPage: 'GeneVisualization'
    };
  },
  methods: {
    showPage(page) {
      this.currentPage = page;
    }
  },
  components: {
    GeneVisualization,
    SummaryReport,
    ModelTrain,
    ModelManage,
    KnowledgeManage
  }
};
</script>

<style scoped>
/* 全局样式 */
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: 'Segoe UI', 'Arial', sans-serif;
  background: #f5f7fa;
  color: #2c3e50;
  line-height: 1.6;
}

#app {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

/* 美化后的头部样式 */
.header {
  background: linear-gradient(135deg, #1e3c6e 0%, #2a5190 100%);
  color: white;
  padding: 2.2rem 1rem;
  text-align: center;
  position: relative;
  overflow: hidden;
  box-shadow: 0 8px 32px rgba(30,60,110,0.2);
  border-bottom-left-radius: 22px;
  border-bottom-right-radius: 22px;
}

.header h1 {
  font-family: 'Segoe UI', 'Arial', sans-serif;
  font-size: 3.5rem;
  margin-bottom: 1rem;
  color: #fff;
  text-shadow: 0 0 15px rgba(255, 255, 255, 0.7), 0 2px 5px rgba(0, 0, 0, 0.15);
  animation: fadeInDown 1s ease;
  font-weight: 700;
  letter-spacing: 1px;
}

.header p {
  font-family: 'Segoe UI', 'Arial', sans-serif;
  color: rgba(255, 255, 255, 0.95);
  font-size: 1.13rem;
  margin-bottom: 0.5rem;
  letter-spacing: 0.5px;
  animation: fadeInUp 1s ease;
  font-weight: 400;
  text-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

/* 模拟生物细胞的背景元素 */
.header::before {
  content: '';
  position: absolute;
  top: -50%;
  left: -50%;
  width: 200%;
  height: 800%;
  background: radial-gradient(circle, rgba(255, 255, 255, 0.15) 10%, transparent 10%);
  background-size: 20px 20px;
  z-index: 0;
  pointer-events: none;
  animation: rotate 40s linear infinite;
  opacity: 0.8;
}

@keyframes fadeInDown {
  from {
    opacity: 0;
    transform: translateY(-20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes rotate {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

.main-container {
  display: flex;
  flex: 1;
  overflow: hidden;
  margin: 15px;
  border-radius: 18px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.08);
  background: white;
}

.sidebar {
  width: 230px;
  background: linear-gradient(180deg, #1e3c6e 0%, #27487e 100%);
  padding: 2rem 1.2rem;
  display: flex;
  flex-direction: column;
  gap: 1.2rem;
  border-top-left-radius: 18px;
  border-bottom-left-radius: 18px;
  box-shadow: 2px 0 15px rgba(0,0,0,0.1);
}

.sidebar button {
  display: flex;
  align-items: center;
  gap: 0.9rem;
  padding: 1rem 1.2rem;
  background-color: transparent;
  color: white;
  border: none;
  border-radius: 10px;
  cursor: pointer;
  transition: all 0.3s ease;
  font-size: 1.05rem;
  text-align: left;
  font-weight: 500;
  letter-spacing: 0.3px;
}

.sidebar button:hover {
  background-color: rgba(42, 81, 144, 0.75);
  transform: translateX(5px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
}

.sidebar button.active {
  background-color: #3a6cb9;
  color: white;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
  font-weight: 600;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
}

.sidebar button i {
  font-size: 1.3rem;
  color: rgba(255, 255, 255, 0.95);
  transition: all 0.3s ease;
}

.sidebar button:hover i {
  transform: scale(1.1);
}

.content {
  flex: 1;
  padding: 2rem;
  overflow-y: auto;
  background: #ffffff;
  border-top-right-radius: 18px;
  border-bottom-right-radius: 18px;
}

/* 过渡效果 */
.fade-enter-active,
.fade-leave-active {
  transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
}

.fade-enter {
  opacity: 0;
  transform: translateY(10px);
}

.fade-leave-to {
  opacity: 0;
  transform: translateY(-10px);
}

.icon-dna:before {
  font-family: 'iconfont';
  content: '\e61a';
  color: #2a5190;
}

.icon-report:before {
  font-family: 'iconfont';
  content: '\e60a';
  color: #2a5190;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .main-container {
    flex-direction: column;
    margin: 10px;
  }
  
  .sidebar {
    width: 100%;
    border-radius: 18px 18px 0 0;
    padding: 1rem;
  }
  
  .content {
    border-radius: 0 0 18px 18px;
    padding: 1.5rem;
  }
  
  .header h1 {
    font-size: 2.5rem;
  }
  
  .header p {
    font-size: 1rem;
  }
}
</style>