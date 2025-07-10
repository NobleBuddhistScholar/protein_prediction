#!/bin/bash

data_path=/root/autodl-tmp
model_name=$1
# nucleotide-transformer-500m-1000g
# nucleotide-transformer-500m-human-ref
model_path=/root/autodl-tmp/Models/$model_name

echo "The provided kmer is: $kmer, data_path is $data_path, model_path is $model_path"
export HF_ENDPOINT=https://hf-mirror.com
# run_name=$model_name
# sh scripts/run_nt_demo.sh nucleotide-transformer-500m-1000g
for seed in 42
do
   for data in prom_300_all prom_300_notata
    do
        python train.py \
            --model_name_or_path $model_path \
            --data_path  ${data_path}/GUE/prom/$data \
            --kmer -1 \
            --run_name ${model_name}_prom_${data}_seed${seed} \
            --model_max_length 70 \
            --use_lora \
            --lora_target_modules 'query,value,key,dense' \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 64 \
            --gradient_accumulation_steps 1 \
            --lora_alpha 16 \
            --learning_rate 1e-4 \
            --num_train_epochs 4 \
            --fp16 \
            --save_steps 400 \
            --output_dir output/${model_name} \
            --eval_strategy no \
            --eval_steps 400 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done
    # for data in covid
    # do
    #     python train.py \
    #         --model_name_or_path $model_path \
    #         --data_path  ${data_path}/GUE/virus/$data \
    #         --kmer -1 \
    #         --run_name ${model_name}_virus_${data}_seed${seed} \
    #         --model_max_length 256 \
    #         --use_lora \
    #         --lora_target_modules 'query,value,key,dense' \
    #         --per_device_train_batch_size 32 \
    #         --per_device_eval_batch_size 32 \
    #         --gradient_accumulation_steps 4 \
    #         --learning_rate 1e-4 \
    #         --num_train_epochs 3 \
    #         --fp16 \
    #         --save_steps 10000 \
    #         --output_dir temp_output/${model_name} \
    #         --eval_strategy no \
    #         --eval_steps 200 \
    #         --warmup_steps 200 \
    #         --logging_steps 100000 \
    #         --overwrite_output_dir True \
    #         --log_level info \
    #         --seed ${seed} \
    #         --find_unused_parameters False
    # done
    # for data in H3
    # do
    #     python train.py \
    #         --model_name_or_path $model_path \
    #         --data_path  ${data_path}/GUE/EMP/$data \
    #         --kmer -1 \
    #         --run_name ${model_name}_EMP_${data}_seed${seed} \
    #         --model_max_length 100 \
    #         --use_lora \
    #         --lora_target_modules 'query,value,key,dense' \
    #         --per_device_train_batch_size 32 \
    #         --per_device_eval_batch_size 64 \
    #         --gradient_accumulation_steps 1 \
    #         --lora_alpha 16 \
    #         --learning_rate 1e-4 \
    #         --num_train_epochs 2 \
    #         --fp16 \
    #         --save_steps 200 \
    #         --output_dir temp_output/${model_name} \
    #         --eval_strategy no \
    #         --eval_steps 200 \
    #         --warmup_steps 50 \
    #         --logging_steps 100000 \
    #         --overwrite_output_dir True \
    #         --log_level info \
    #         --seed ${seed} \
    #         --find_unused_parameters False
    # done
done