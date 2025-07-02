// 如果您需要使用 Vue Router，可以在这里设置路由
import Vue from 'vue';
import Router from 'vue-router';
import GeneVisualization from './components/GeneVisualization.vue';
import SummaryReport from './components/SummaryReport.vue';

Vue.use(Router);

export default new Router({
  routes: [
    {
      path: '/gene-visualization',
      component: GeneVisualization
    },
    {
      path: '/summary-report',
      component: SummaryReport
    }
  ]
});
