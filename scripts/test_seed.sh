#!/bin/bash

data_path=/root/autodl-tmp
model_name=$1

model_path=/root/autodl-tmp/Models/$model_name

echo "The provided kmer is: $kmer, data_path is $data_path, model_path is $model_path"
export HF_ENDPOINT=https://hf-mirror.com

for seed in 42
do
    for data in H4 H3
    do
        python train.py \
            --model_name_or_path $model_path \
            --data_path  ${data_path}/GUE/EMP/$data \
            --kmer -1 \
            --run_name ${model_name}_EMP_${data}_seed${seed} \
            --model_max_length 100 \
            --use_lora \
            --lora_target_modules 'q_proj,v_proj,k_proj,o_proj' \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 64 \
            --gradient_accumulation_steps 1 \
            --lora_alpha 16 \
            --learning_rate 1e-4 \
            --num_train_epochs 3 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/${model_name}_seed${seed} \
            --eval_strategy no \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done
done