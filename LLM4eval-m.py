import os
import json
import logging
import time
import datetime
import base64
import hmac
import hashlib
import requests
import random
import uuid
import pandas as pd
import re
import tqdm
import csv
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from tqdm.contrib.concurrent import process_map
from multiprocessing import Manager, Process
from functools import partial

prompt1 = """
        You are good at TextQA, capable of answering user questions based on provided textual content. The user will supply a piece of text, which is a caption describing the content of a video or image. Your task is to answer the user's question based on this caption. Please respond using the following format: 
            Answer@@@Detailed Analysis

        Format Analysis:
            Answer: Indicate whether the answer to the user's question is "Yes" "No" or "Unanswerable",
            Detailed Analysis: Provide a detailed analysis supporting your conclusion.
            If the caption does not contain enough information to answer the user's question, respond with "Unanswerable" and provide an explanation for your inability to answer.

        Example:
            Caption: "A group of people are playing soccer in a park on a sunny day."
            User's Question: "Is it raining in the video?"
            Response must:
                No@@@The caption describes the scene as taking place on a sunny day, which implies that it is not raining.

        Your Task:
        The caption for the video/image is as follows: "{}"
        The user's question is: "{}"

        Please ensure that your response adheres to the structured format provided, as "Answer@@@Detailed Analysis" (The answer can only be one of "Yes", "No" or "Unanswerable"). Don't use any prefixes, just answer directly.
        """

def load_evalset(file_path):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            data[json_obj["video_id"]]  = json_obj       
    return data


def process_row(row, qa_dat, caption_col, apis):
    results = []
    eval_caption = row[caption_col]
    if row["video_id"] in qa_dat:
        qa_pairs = qa_dat[row["video_id"]]["qa"]
    else:
        return results

    for qa in qa_pairs:
        try:
            question = qa.split("*")[1]
            answer = qa.split("*")[2]
        except Exception as e:
            print(qa)
            print(row["video_id"])
            continue
        prompt = prompt1.format(eval_caption, question)
        for _ in range(15):
            try:
                api = random.choice(apis)
                res = api.call_data_eval(prompt)
                res = json.dumps(res.json(), indent=2, ensure_ascii=False)
                res = json.loads(res)
                # print(res)
                ans, reason = res["answer"][0]["value"].split("@@@")
                break
            except Exception as e:
                # print(e)
                # print(res)
                # print(res["answer"][0]["value"])
                time.sleep(1)
                continue
        try:
            if ans.lower() in ["yes", "no"] or answer.lower() == ans.lower():
                ans_stat = int(answer.lower() == ans.lower())
            elif ans.lower() == "unanswerable":
                ans_stat = 2
            else:
                continue
        except Exception as e:
            print(e)
            continue
        
        results.append((row["video_id"], row["cos_url"], eval_caption, qa, ans_stat, f"{ans}, {reason}"))

    # print(results[-1])
    return results


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='input CSV file')
    parser.add_argument('--output_dir', type=str, default=None)
    # 加载评测caption
    parser.add_argument('--caption_col', type=str, default="HYCaptioner")
    parser.add_argument('--llm', type=str, default="gpt4o")
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument('--dataset_path', type=str, default="testing_5k.jsonl")
    return parser.parse_args()


if __name__ == "__main__":
    
    args = parse_arguments()

    if args.llm == "gpt4o":
        print("gpt4o start")
        gapis = [gpt4o_api]
    else:
        gapis = [gemini_api]



    ### load dataset 
    qa_dat = load_evalset(args.dataset_path)
    # 定义文件路径
    df = pd.read_csv(args.input_file, index_col=None, sep=',')

    # 使用多进程处理
    # with Manager() as manager:
    # Define the partial function with the necessary arguments
    partial_process_row = partial(process_row, qa_dat=qa_dat, caption_col=args.caption_col, apis=gapis)

    # Use process_map with the specified chunksize
    results = process_map(partial_process_row, [row for _, row in df.iterrows()], max_workers=args.max_workers, chunksize=4)

    # 打开CSV文件写入结果
    if args.output_dir is not None:
        with open(os.path.join(args.output_dir, f"{args.llm}_eval_{args.caption_col}.csv"), mode='w', encoding='utf-8-sig', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["video_id", 'cos_url', "caption", "question", "answer", "analysis"])
            ans_stats = []
            for result in results:
                for row in result:
                    writer.writerow(row)
                    ans_stats.append(row[4])
    else:
        ans_stats = []
        for result in results:
            for row in result:
                ans_stats.append(row[4])
    # 计算总数
    total = len(ans_stats)

    # 计算每种结果的数量
    correct_count = ans_stats.count(1)
    incorrect_count = ans_stats.count(0)
    uncertain_count = ans_stats.count(2)

    # 计算比例
    accuracy = correct_count / total
    error_rate = incorrect_count / (total-uncertain_count)
    uncertainty_rate = 1- uncertain_count / total

    # 输出结果
    print(f"AR: {accuracy:.2%}")
    print(f"IR: {error_rate:.2%}")
    print(f"CR: {uncertainty_rate:.2%}")

    # 打开文件以写入
    with open('results.txt', 'a+') as file:
        # 获取当前时间
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # 写入文件
        file.write(f"{args.llm}_{args.caption_col}-{current_time} - AR: {accuracy:.2%}, IR: {error_rate:.2%}, CR: {uncertainty_rate:.2%}\n")

    print("结果已写入 results.txt 文件。")

    






    
    
    
    
    
    

   
    
    
    
    
