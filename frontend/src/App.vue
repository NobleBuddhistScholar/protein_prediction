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
        <button @click="showPage('SummaryReport')" :class="{ active: currentPage === 'SummaryReport' }">
          <i class="fa-solid fa-folder-open"></i>  总报告管理
        </button>
        <button @click="showPage('ModelTrain')" :class="{ active: currentPage === 'ModelTrain' }">
          <i class="fa-solid fa-folder-open"></i>  模型训练
        </button>
        <button @click="showPage('ModelManage')" :class="{ active: currentPage === 'ModelManage' }">
          <i class="fa-solid fa-folder-open"></i>  模型管理
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
    ModelManage
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
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  background-color: #92afcd;
  color: #333;
  line-height: 1.6;
}

#app {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

/* 美化后的头部样式 */
.header {
  background: linear-gradient(135deg, #33465a 0%, #8eaee5 100%);
  color: white;
  padding: 2rem 1rem;
  text-align: center;
  position: relative;
  overflow: hidden;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  border-bottom-left-radius: 20px;
  border-bottom-right-radius: 20px;
}

.header h1 {
  font-family: 'Dancing Script', cursive;
  font-size: 4rem;
  margin-bottom: 1rem;
  color: #fff;
  text-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
  animation: fadeInDown 1s ease;
}

.header p {
  font-family: 'Open Sans', sans-serif;
  color: #ecf0f1;
  font-size: 1.2rem;
  margin-bottom: 0.5rem;
  letter-spacing: 0.05em;
  animation: fadeInUp 1s ease;
}

/* 模拟生物细胞的背景元素 */
.header::before {
  content: '';
  position: absolute;
  top: -50%;
  left: -50%;
  width: 200%;
  height: 800%;
  background: radial-gradient(circle, rgba(255, 255, 255, 0.2) 15%, transparent 15%);
  background-size: 25px 20px;
  z-index: 0;
  pointer-events: none;
  animation: rotate 30s linear infinite;
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
  margin: 10px;
  border-radius: 20px;
  box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
}

.sidebar {
  width: 220px;
  background: linear-gradient(180deg, #567798 0%, #506f8e 100%);
  padding: 1.5rem 1rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
  border-top-left-radius: 20px;
  border-bottom-left-radius: 20px;
}

.sidebar button {
  display: flex;
  align-items: center;
  gap: 0.8rem;
  padding: 0.8rem 1rem;
  background-color: transparent;
  color: #eceef1;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.3s ease;
  font-size: 1rem;
  text-align: left;
}

.sidebar button:hover {
  background-color: rgba(255, 255, 255, 0.1);
  transform: translateX(5px);
}

.sidebar button.active {
  background-color: rgba(255, 255, 255, 0.3);
  color: white;
  box-shadow: 0 0 5px rgba(255, 255, 255, 0.5);
}

.sidebar button i {
  font-size: 1.2rem;
}

.content {
  flex: 1;
  padding: 2rem;
  overflow-y: auto;
  background-color: rgb(206, 215, 237);
  border-top-right-radius: 20px;
  border-bottom-right-radius: 20px;
}

/* 过渡效果 */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter,
.fade-leave-to {
  opacity: 0;
}

.icon-dna:before {
  font-family: 'iconfont';
  content: '\e61a';
}

.icon-report:before {
  font-family: 'iconfont';
  content: '\e60a';
}
</style>    