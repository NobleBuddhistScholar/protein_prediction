# #!/bin/bash

# # 设置日志文件名
# LOG_FILE="dnabert_all_run.log"

# # 输出开始信息
# echo "=== DNABERT batch run started at $(date) ===" >> $LOG_FILE

# # 定义参数列表
# for i in 3 4 5 6
# do
#   echo "Starting run_dnabert1_faster.sh $i at $(date)" >> $LOG_FILE
#   sh scripts/run_dnabert1_faster.sh $i >> $LOG_FILE 2>&1
#   echo "Finished run_dnabert1_faster.sh $i at $(date)" >> $LOG_FILE
# done

# echo "=== DNABERT batch run finished at $(date) ===" >> $LOG_FILE

#!/bin/bash

LOG_FILE="dnabert_all_run.log"
echo "=== DNABERT batch run started at $(date) ===" >> $LOG_FILE

mkdir -p logs

# 定义参数列表
for i in 3 4 5 6
do
  echo "Starting run_dnabert1_faster.sh $i at $(date)" >> $LOG_FILE

  TASK_LOG="logs/dnabert_kmer${i}_run.log"
  bash scripts/run_dnabert1_faster.sh $i >> $TASK_LOG 2>&1

  if [ $? -eq 0 ]; then
    echo "Finished run_dnabert1_faster.sh $i successfully at $(date)" >> $LOG_FILE
  else
    echo "ERROR during run_dnabert1_faster.sh $i at $(date)" >> $LOG_FILE
  fi
done

echo "=== DNABERT batch run finished at $(date) ===" >> $LOG_FILE
