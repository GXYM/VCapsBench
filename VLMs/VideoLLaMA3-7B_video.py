import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from prompts import prompt
import csv
import pandas as pd
import time


device = "cuda:0"
model_path = "VideoLLaMA3-7B"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map={"": device},
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)


if __name__ == "__main__":

    video_dir =  "VCapsBench/videos5k"
    # 定义文件路径
    # 读取 CSV 文件
    input_fn = "VCapsbench_Caption_ALL.csv"
    df = pd.read_csv(input_fn, index_col=None, sep=',')

    output_file = "VCapsbench_Caption_ALL_add_VideoLLaMA3.csv"

    # 读取已有的 CSV 文件
    try:
        existing_df = pd.read_csv(output_file, index_col=None, sep=',')
        existing_video_ids = set(existing_df['video_id'].values)
    except FileNotFoundError:
        existing_df = pd.DataFrame()
        existing_video_ids = set()
    
     # 过滤掉已经存在的数据
    new_rows = df[~df['video_id'].isin(existing_video_ids)].to_dict('records')
    # 检查输出文件是否存在，如果不存在则创建并写入表头
    if existing_df.empty:
        with open(output_file, mode='w', encoding='utf-8-sig', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["video_id", "source", "resolution", "duration", "content_fine_category", "content_parent_category", 'video_path', 
            "VideoLLaMA3-7B_caption"])
    
    for index, row in enumerate(new_rows):
        video_path = os.path.join(video_dir, row["video_id"])
        print(index, len(new_rows), video_path)

        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": {"video_path": video_path, "fps": 2, "max_frames": 180}},
                    {"type": "text", "text": prompt},
                ]
            },
        ]

        inputs = processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        t0 = time.time()
        for _ in range(10):
            try:
                output_ids = model.generate(**inputs, max_new_tokens=512)
                response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                print(response)
                print(f'video id: {row["video_id"]}\nAssistant: {response}')
            except Exception as e:
                print(e)
                continue
            if len(response) > 10:
                break

        with open(output_file, mode='a', encoding='utf-8-sig', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([row['video_id'], row['source'], row['resolution'], row['duration'], row['content_fine_category'], row['content_parent_category'], response])
        
        print(f"time cost: {time.time()- t0}")
