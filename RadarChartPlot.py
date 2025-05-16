import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import defaultdict

# 配置字体
FONT_PATH = 'times.ttf'
PROP_TITLE = fm.FontProperties(fname=FONT_PATH, weight='bold', size=20)
PROP_LABEL = fm.FontProperties(fname=FONT_PATH, weight='bold', size=12)

# 类别顺序
CATEGORY_ORDER = [
    'proper noun', 'action', 'relation', 'count', 'entity size', 'entity', 'entity shape',
    'lighting', 'color palette', 'color grading', 'color', 'position', 'relative position',
    'background', 'text', 'blur', 'style', 'camera movement', 'shot type', 'emotion', 'atmosphere'
]

def load_csv(file_path):
    """读取csv，返回有序字典：{类别: [答案列表]}"""
    q_dic = defaultdict(list)
    df = pd.read_csv(file_path)
    for _, row in df.iterrows():
        for qa in row["question"].split("*"):
            q_dic[qa.strip().lower()].append(row["answer"])
    # 按指定顺序排序
    return {k: q_dic[k] for k in CATEGORY_ORDER if k in q_dic}

def calculate_ratio(dic, var=1):
    """计算每个类别的var比例"""
    result = {}
    for k, v in dic.items():
        v = [int(x) for x in v if str(x).isdigit()]
        if not v:
            result[k] = 0.0
            continue
        if var == 2:
            result[k] = 1 - (v.count(var) / len(v))
        elif var == 0:
            denom = len(v) - v.count(2)
            result[k] = v.count(var) / denom if denom else 0.0
        else:
            result[k] = v.count(var) / len(v)
    return result

def calculate_total_metrics(dic, var=1):
    """计算所有类别合并后的总指标"""
    all_answers = []
    for v in dic.values():
        all_answers.extend([int(x) for x in v if str(x).isdigit()])
    if not all_answers:
        return 0.0
    if var == 2:
        return 1 - (all_answers.count(var) / len(all_answers))
    elif var == 0:
        denom = len(all_answers) - all_answers.count(2)
        return all_answers.count(var) / denom if denom else 0.0
    else:
        return all_answers.count(var) / len(all_answers)

def create_radar_chart(dic, color, ax):
    """绘制单个模型的雷达图"""
    labels = np.array(list(dic.keys()))
    stats = np.array(list(dic.values()))
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    stats = np.concatenate((stats, [stats[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    labels = np.concatenate((labels, [labels[0]]))
    ax.plot(angles, stats, color=color)
    ax.set_yticklabels([])
    ax.set_thetagrids(np.degrees(angles), labels, fontproperties=PROP_LABEL)

def main():
    datas = {
        "NVILA-8B": "gpt4o_eval_VILA-8B_caption.csv",
        "InternVL2.5-8B": "gpt4o_eval_InternVL2.5-8B_caption.csv",
        "LLaVA-Video-7B": "gpt4o_eval_LLaVA-Video-7B.csv",
        "Qwen2VL-7B": "gpt4o_eval_qwen2vl7B_caption.csv",
        "VideoLLaMA3-7B": "gpt4o_eval_VideoLLaMA3-7B_caption.csv",
        "Qwen2.5-VL-7B": "gpt4o_eval_Qwen2.5-VL-7B.csv",
        "Qwen2.5-VL-72B": "gpt4o_eval_Qwen2.5-VL-72B.csv",
        "GPT-4o": "gpt4o_eval_gpt4o_cap.csv",
        "Gemini2.5-Pro-Flash": "gpt4o_eval_gemini2.5_pre_flash.csv",
        "Gemini-2.5-Pro-Preview": "gpt4o_eval_gemini2.5_pro-05-06.csv",
       
    }
    color_dic = {
        "NVILA-8B": "limegreen", 
        "InternVL2.5-8B": "orange", 
        "LLaVA-Video-7B": "hotpink",
        "Qwen2VL-7B": "aqua", 
        "VideoLLaMA3-7B": "mediumspringgreen", 
        "Qwen2.5-VL-7B": "dodgerblue", 
        "Qwen2.5-VL-72B": "navajowhite", 
        "GPT-4o": "chocolate",
        "Gemini2.5-Pro-Flash": "aquamarine" ,
        "Gemini-2.5-Pro-Preview": "deeppink"
    }
     # 
     # 'red', green, yellow, indigo, orangered, perrywinkle, pinkred, violet, neonyellow, orangish, deeppink, lawngreen, chocolate, aquamarine
    metric_names = {1: "Accuracy (AR)", 0: "Inconsistency Rate (IR)", 2: "Coverage Rate (CR)"}

    # 读取数据
    model_data = {name: load_csv(path) for name, path in datas.items()}

    # 绘制雷达图
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))
    for i, (var, metric_name) in enumerate(metric_names.items()):
        labels = []
        for name, data in model_data.items():
            labels.append(name)
            dic_rate = calculate_ratio(data, var)
            create_radar_chart(dic_rate, color_dic[name], axs[i])
            # print(name, dic_rate)
            # print(name, metric_name, " &".join([str(round(v*100, 2)) for v in dic_rate.values()]))

        #     # 假设 dic_rate.values() 已经有21个元素
        #     values = list(dic_rate.values())
        #     splits = [7, 4, 5, 5]
        #     result = []
        #     idx = 0

        #     for split in splits:
        #         for jj in range(split):
        #             v = round(values[idx]*100, 2)
        #             if jj == split - 1:
        #                 # 最后一个元素用 multicolumn
        #                 result.append(f"\\multicolumn{{1}}{{c|}}{{{v}}}")
        #             else:
        #                 result.append(str(v))
        #             idx += 1

        #     # 拼接
        #     output = f"{name} & {metric_name} & " + " & ".join(result)
        #     print(output)
        #     print("\n")
        # print("\n\n")
        axs[i].set_title(metric_name, fontproperties=PROP_TITLE)
        axs[i].legend(labels, loc='lower center', bbox_to_anchor=(0.5, -0.40), ncol=2, prop=PROP_LABEL)
    plt.tight_layout()
    plt.savefig('radar_chart_all-New2.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    # 打印总指标
    print("\n==== Total Metrics for Each Model ====")
    header = ["Model", "Accuracy (AR)", "Inconsistency Rate (IR)", "Coverage Rate (CR)"]
    print("{:<20} {:>15} {:>20} {:>20}".format(*header))
    for name, data in model_data.items():
        acc = calculate_total_metrics(data, 1)
        irr = calculate_total_metrics(data, 0)
        cov = calculate_total_metrics(data, 2)
        print("{:<20} {:>15.4f} {:>20.4f} {:>20.4f}".format(name, acc, irr, cov))

if __name__ == "__main__":
    main()