from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from prompts import prompt
import csv
import pandas as pd
import time
import os

# default: Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
# )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained("Qwen2.5-VL-7B-Instruct")



if __name__ == "__main__":

    video_dir =  "VCapsBench/videos5k"
    # 定义文件路径
    # 读取 CSV 文件
    input_fn = "VCapsbench_Caption_ALL.csv"
    df = pd.read_csv(input_fn, index_col=None, sep=',')

    output_file = "VCapsbench_Caption_ALL_add_Qwen2.5VL_7B.csv"

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
        print("hhhhhhh")
        with open(output_file, mode='w', encoding='utf-8-sig', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["video_id", "source", "resolution", "duration", "content_fine_category", "content_parent_category", "Qwen2.5-VL-7B"])
    
    for index, row in enumerate(new_rows):
        video_path = os.path.join(video_dir, row["video_id"])
        print(index, len(new_rows), video_path)

        # Messages containing a local video path and a text query
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        #In Qwen 2.5 VL, frame rate information is also input into the model to align with absolute time.
        # Preparation for inference
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            fps=2,
            padding=True,
            return_tensors="pt",
            # **video_kwargs,
        )
        inputs = inputs.to("cuda")
        t0 = time.time()
        for _ in range(10):
            try:
                # Inference
                generated_ids = model.generate(**inputs, max_new_tokens=1024)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                print(f'video id: {row["video_id"]}\nAssistant: {output_text[0]}')
            except Exception as e:
                print(e)
                continue
            if len(output_text[0]) > 10:
                break

        with open(output_file, mode='a', encoding='utf-8-sig', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([row['video_id'], row['source'], row['resolution'], row['duration'], row['content_fine_category'], row['content_parent_category'], output_text[0]])
        
        print(f"time cost: {time.time()- t0}")