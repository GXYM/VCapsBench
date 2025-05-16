# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import torch
import sys
import warnings
from decord import VideoReader, cpu
import numpy as np
from prompts import prompt
import csv
import pandas as pd
import time
import os
warnings.filterwarnings("ignore")


def load_video(video_path, max_frames_num,fps=1,force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    return spare_frames,frame_time,video_time
pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
# device_map = "cuda:1"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
model.eval()
max_frames_num = 32


if __name__ == "__main__":

    video_dir =  "VCapsBench/videos5k"
    # 定义文件路径
    # 读取 CSV 文件
    input_fn = "VCapsBench_Caption_ALL.csv"
    df = pd.read_csv(input_fn, index_col=None, sep=',')

    output_file = "VCapsBench_Caption_ALLl_add_lavavideo.csv"

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
            writer.writerow(["video_id", "source", "resolution", "duration", "content_fine_category", "content_parent_category", "LLaVA-Video-7B"])
    
    for index, row in enumerate(new_rows):
        video_path = os.path.join(video_dir, row["video_id"])
        print(index, video_path)

        video,frame_time,video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
        video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().to(torch.bfloat16)
        video = [video]
        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
        time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
        question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruciton}\n{prompt}"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        t0 = time.time()
        for _ in range(10):
            # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
            try:
                cont = model.generate(
                    input_ids,
                    images=video,
                    modalities= ["video"],
                    do_sample=False,
                    temperature=0.1,
                    max_new_tokens=4096,
                )
                text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
                print(f'video id: {row["video_id"]}\nAssistant: {text_outputs}')
            except Exception as e:
                print(e)
                continue
            if len(text_outputs) > 10:
                break

        with open(output_file, mode='a', encoding='utf-8-sig', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([row['video_id'], row['source'], row['resolution'], row['duration'], row['content_fine_category'], row['content_parent_category'], text_outputs])
        
        print(f"time cost: {time.time()- t0}")


