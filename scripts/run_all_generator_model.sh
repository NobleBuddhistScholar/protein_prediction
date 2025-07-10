LOG_FILE="generator_all_run_with_seed.log"
echo "=== generator batch run started at $(date) ===" >> $LOG_FILE

mkdir -p logs


for model_name in GENERator-eukaryote-1.2b-base
do
  echo "Starting $model_name at $(date)" >> $LOG_FILE

  MODEL_LOG="logs/${model_name}_run_with_seed.log"
  bash scripts/run_generator_faster.sh $model_name >> $MODEL_LOG 2>&1

  if [ $? -eq 0 ]; then
    echo "Finished $model_name successfully at $(date)" >> $LOG_FILE
  else
    echo "ERROR during $model_name at $(date)" >> $LOG_FILE
  fi
done

echo "=== generator batch run finished at $(date) ===" >> $LOG_FILE
