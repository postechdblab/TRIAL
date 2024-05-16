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
    echo "CUDA_VISIBLE_DEVICES=$device python scripts/analysis/token_scores_from_cache.py --dataset $dataset --file_path /root/ColBERT/debug/result.${dataset}_baseline_nway32_q4_less_hard_lr2_distill.pkl --q_max_len 64 --d_max_len 300"
    CUDA_VISIBLE_DEVICES=$device python scripts/analysis/token_scores_from_cache.py --dataset $dataset --file_path /root/ColBERT/debug/result.${dataset}_baseline_nway32_q4_less_hard_lr2_distill.pkl --q_max_len 64 --d_max_len 300
done