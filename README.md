# VCapsBench
VCapsBench: A Large-scale Fine-grained Benchmark for Video Caption Quality Evaluation  

## VCapsBench Data
```
Huggingface: somos99/VCapsBench
```
## Video Caption Generate Script in VLMs
* Qwen2.5-VL-72B
* Qwen2.5-VL-7B 
* Qwen2VL-7B
* InternVL2.5-8B
* NVILA-8B
* LLaVA-Video-7B 
* VideoLLaMA3-7B

## Video Caption Eval Script
* LLM4eval-m.py
* eval.sh
```
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
```
##  Evaluation Result Visualization Script
* RadarChartPlot.py
  ![](https://github.com/GXYM/VCapsBench/blob/main/imgs/iShot_2025-05-16_19.23.55.png)
* WordLength.py
  ![](https://github.com/GXYM/VCapsBench/blob/main/imgs/iShot_2025-05-16_19.24.42.png)
* wordlength_IR_CR_plot.py
  ![](https://github.com/GXYM/VCapsBench/blob/main/imgs/iShot_2025-05-16_19.24.18.png)





