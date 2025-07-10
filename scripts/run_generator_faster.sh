#!/bin/bash

data_path=/root/autodl-tmp
model_name=$1

model_path=/root/autodl-tmp/Models/$model_name

echo "The provided kmer is: $kmer, data_path is $data_path, model_path is $model_path"
export HF_ENDPOINT=https://hf-mirror.com



for seed in 42
do
    for data in H3 H3K14ac H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K79me3 H3K9ac H4 H4ac
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
            --output_dir output/${model_name} \
            --eval_strategy no \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done



    for data in prom_core_all prom_core_notata
    do
        python train.py \
            --model_name_or_path $model_path \
            --data_path  ${data_path}/GUE/prom/$data \
            --kmer -1 \
            --run_name ${model_name}_prom_${data}_seed${seed} \
            --model_max_length 20 \
            --use_lora \
            --lora_target_modules 'q_proj,v_proj,k_proj,o_proj' \
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


    for data in prom_core_tata
    do
        python train.py \
            --model_name_or_path $model_path \
            --data_path  ${data_path}/GUE/prom/$data \
            --kmer -1 \
            --run_name ${model_name}_prom_${data}_seed${seed} \
            --model_max_length 20 \
            --use_lora \
            --lora_target_modules 'q_proj,v_proj,k_proj,o_proj' \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 64 \
            --gradient_accumulation_steps 1 \
            --lora_alpha 16 \
            --learning_rate 1e-4 \
            --num_train_epochs 10 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/${model_name} \
            --eval_strategy no \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done

    for data in prom_300_all prom_300_notata
    do
        python train.py \
            --model_name_or_path $model_path \
            --data_path  ${data_path}/GUE/prom/$data \
            --kmer -1 \
            --run_name ${model_name}_prom_${data}_seed${seed} \
            --model_max_length 70 \
            --use_lora \
            --lora_target_modules 'q_proj,v_proj,k_proj,o_proj' \
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


    for data in prom_300_tata
    do
        python train.py \
            --model_name_or_path $model_path \
            --data_path  ${data_path}/GUE/prom/$data \
            --kmer -1 \
            --run_name ${model_name}_prom_${data}_seed${seed} \
            --model_max_length 70 \
            --use_lora \
            --lora_target_modules 'q_proj,v_proj,k_proj,o_proj' \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 64 \
            --gradient_accumulation_steps 1 \
            --lora_alpha 16 \
            --learning_rate 1e-4 \
            --num_train_epochs 10 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/${model_name} \
            --eval_strategy no \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done


    for data in reconstructed
    do
        python train.py \
            --model_name_or_path $model_path \
            --data_path  ${data_path}/GUE/splice/$data \
            --kmer -1 \
            --run_name ${model_name}_splice_${data}_seed${seed} \
            --model_max_length 80 \
            --use_lora \
            --lora_target_modules 'q_proj,v_proj,k_proj,o_proj' \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 64 \
            --gradient_accumulation_steps 1 \
            --lora_alpha 16 \
            --learning_rate 1e-4 \
            --num_train_epochs 5 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/${model_name} \
            --eval_strategy no \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done



    for data in covid
    do
        python train.py \
            --model_name_or_path $model_path \
            --data_path  ${data_path}/GUE/virus/$data \
            --kmer -1 \
            --run_name ${model_name}_virus_${data}_seed${seed} \
            --model_max_length 256 \
            --use_lora \
            --lora_target_modules 'q_proj,v_proj,k_proj,o_proj' \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --gradient_accumulation_steps 4 \
            --learning_rate 1e-4 \
            --num_train_epochs 3 \
            --fp16 \
            --save_steps 10000 \
            --output_dir output/${model_name} \
            --eval_strategy no \
            --eval_steps 200 \
            --warmup_steps 200 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done

    for data in 0 1 2 3 4
    do 
        python train.py \
            --model_name_or_path $model_path \
            --data_path  ${data_path}/GUE/mouse/$data \
            --kmer -1 \
            --run_name ${model_name}_mouse_${data}_seed${seed} \
            --model_max_length 30 \
            --use_lora \
            --lora_target_modules 'q_proj,v_proj,k_proj,o_proj' \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 64 \
            --gradient_accumulation_steps 1 \
            --lora_alpha 16 \
            --learning_rate 1e-4 \
            --num_train_epochs 5 \
            --max_steps 1000 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/${model_name} \
            --eval_strategy no \
            --eval_steps 200 \
            --warmup_steps 100 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done


    for data in 0 1 2 3 4
    do 
        python train.py \
            --model_name_or_path $model_path \
            --data_path  ${data_path}/GUE/tf/$data \
            --kmer -1 \
            --run_name ${model_name}_tf_${data}_seed${seed} \
            --model_max_length 30 \
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
            --output_dir output/${model_name} \
            --eval_strategy no \
            --eval_steps 200 \
            --warmup_steps 30 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done
done