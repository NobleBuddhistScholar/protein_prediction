# #!/bin/bash

# # 设置日志文件名
# LOG_FILE="gena_lm_all_run.log"

# # 输出开始信息
# echo "=== GENA_LM batch run started at $(date) ===" >> $LOG_FILE

# # 定义参数列表
# # for model_name in gena-lm-bert-base gena-lm-bert-base-lastln-t2t gena-lm-bert-base-t2t gena-lm-bert-base-t2t-multi gena-lm-bigbird-base-t2t
# for model_name in gena-lm-bigbird-base-t2t
# do
#   MODEL_LOG="logs/${model_name}_run.log"
#   mkdir -p logs

#   echo "Starting $model_name at $(date)" >> $LOG_FILE
#   sh scripts/run_gena_lm_faster.sh $model_name >> $MODEL_LOG 2>&1
#   echo "Finished $model_name at $(date)" >> $LOG_FILE
# done

# echo "=== gena-lm batch run finished at $(date) ===" >> $LOG_FILE

#!/bin/bash

LOG_FILE="gena_lm_all_run.log"
echo "=== GENA_LM batch run started at $(date) ===" >> $LOG_FILE

mkdir -p logs

# 如果要跑gena-lm所有的就取消注释，用下面这一行，不然就只跑bigbird
# for model_name in gena-lm-bert-base gena-lm-bert-base-lastln-t2t gena-lm-bert-base-t2t gena-lm-bert-base-t2t-multi gena-lm-bigbird-base-t2t
for model_name in gena-lm-bigbird-base-t2t
do
  echo "Starting $model_name at $(date)" >> $LOG_FILE

  MODEL_LOG="logs/${model_name}_run.log"
  bash scripts/run_gena_lm_faster.sh $model_name >> $MODEL_LOG 2>&1

  if [ $? -eq 0 ]; then
    echo "Finished $model_name successfully at $(date)" >> $LOG_FILE
  else
    echo "ERROR during $model_name at $(date)" >> $LOG_FILE
  fi
done

echo "=== GENA_LM batch run finished at $(date) ===" >> $LOG_FILE

