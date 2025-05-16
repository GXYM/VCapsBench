#!/bin/bash

unset http_proxy      
unset https_proxy

input_file="VCapsBench_Caption_ALL.csv"
dataset_path="VCapsBench_100KQA.jsonl"
max_workers=128
llm="gemini"
# llm="gpt4o"

output_dir="eval_results-gemini-2.5"
caption_cols=("gpt4o_cap" "Qwen2.5-VL-72B" "gemini2.5_pro-05-06" "gemini2.5_pre_flash")


for caption_col in "${caption_cols[@]}"; do
    python3 LLM4eval-m.py \
        --input_file "$input_file" \
        --dataset_path "$dataset_path" \
        --output_dir "$output_dir" \
        --caption_col "$caption_col" \
        --llm "$llm" \
        --max_workers "$max_workers"
done