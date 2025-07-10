LOG_FILE="nt_all_run.log"
echo "=== nt batch run started at $(date) ===" >> $LOG_FILE

mkdir -p logs


for model_name in nucleotide-transformer-500m-1000g nucleotide-transformer-500m-human-ref
do
  echo "Starting $model_name at $(date)" >> $LOG_FILE

  MODEL_LOG="logs/${model_name}_run.log"
  bash scripts/run_nt_faster.sh $model_name >> $MODEL_LOG 2>&1

  if [ $? -eq 0 ]; then
    echo "Finished $model_name successfully at $(date)" >> $LOG_FILE
  else
    echo "ERROR during $model_name at $(date)" >> $LOG_FILE
  fi
done

echo "=== nt batch run finished at $(date) ===" >> $LOG_FILE
