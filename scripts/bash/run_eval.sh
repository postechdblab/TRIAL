#!/bin/bash
device=$1
# Assuming datasets are passed as a space-separated string
datasets=$2

# Split the datasets string into an array using spaces as the delimiter
IFS=' ' read -r -a dataset_array <<< "$datasets"

# Loop over the array of datasets
for dataset in "${dataset_array[@]}"
do
    echo "Processing dataset: $dataset"
    echo "CUDA_VISIBLE_DEVICES=$device python scripts/evaluate/evaluate.py --dataset $dataset --model baseline_nway32_q4_less_hard_lr2_distill --skip_padding --save_result --save_score --include_gold --save_dir /root/EAGLE/debug/ --max_q_length 64 --max_d_length 300"
    CUDA_VISIBLE_DEVICES=$device python scripts/evaluate/evaluate.py --dataset $dataset --model baseline_nway32_q4_less_hard_lr2_distill --skip_padding --save_result --save_score --include_gold --save_dir /root/EAGLE/debug/ --max_q_length 64 --max_d_length 300
done