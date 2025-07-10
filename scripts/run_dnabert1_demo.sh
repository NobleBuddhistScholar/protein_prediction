#!/bin/bash

# This is your argument
data_path=/root/autodl-tmp

kmer=$1

model_path=/root/autodl-tmp/Models/DNA_bert_$kmer

echo "The provided kmer is: $kmer, data_path is $data_path, model_path is $model_path"
export HF_ENDPOINT=https://hf-mirror.com
# sh scripts/run_dna1.sh 3 ; sh scripts/run_dna1.sh 4 ; sh scripts/run_dna1.sh 5 ; sh scripts/run_dna1.sh 6

for seed in 42
do
    # for data in H3
    # do
    #     python train.py \
    #         --model_name_or_path $model_path \
    #         --data_path  ${data_path}/GUE/EMP/$data \
    #         --kmer ${kmer} \
    #         --run_name DNABERT1_${kmer}_EMP_${data}_seed${seed} \
    #         --model_max_length 512 \
    #         --per_device_train_batch_size 32 \
    #         --per_device_eval_batch_size 64 \
    #         --gradient_accumulation_steps 1 \
    #         --learning_rate 3e-5 \
    #         --num_train_epochs 1 \
    #         --fp16 \
    #         --save_steps 200 \
    #         --output_dir temp_output/dnabert1_${kmer} \
    #         --eval_strategy no \
    #         --eval_steps 200 \
    #         --warmup_steps 50 \
    #         --logging_steps 100000 \
    #         --overwrite_output_dir True \
    #         --log_level info \
    #         --seed ${seed} \
    #         --find_unused_parameters False
    # done

    for data in covid
    do
        python train.py \
            --model_name_or_path $model_path \
            --data_path  ${data_path}/GUE/virus/$data \
            --kmer ${kmer} \
            --run_name DNABERT1_${kmer}_virus_${data}_seed${seed} \
            --model_max_length 512 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --gradient_accumulation_steps 4 \
            --learning_rate 3e-5 \
            --num_train_epochs 2 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/dnabert1_${kmer} \
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