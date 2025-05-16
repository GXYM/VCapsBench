import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Path to the font file
font_path = 'times.ttf'
# Create a font object with bold weight and specific size
prop = fm.FontProperties(fname=font_path, weight='bold', size=28)
prop1 = fm.FontProperties(fname=font_path, weight='bold', size=22)
prop2 = fm.FontProperties(fname=font_path, weight='bold', size=24)




# 读取CSV文件
df = pd.read_csv('VCapsBench_Caption_ALL.csv')

# 定义要统计的列名
# 定义要统计的列名
#  'gemini_caption',
columns_to_analyze = [
     'Qwen2.5-VL-7B', 'InternVL2.5-8B_caption',
    'VILA-8B_caption', 'VideoLLaMA3-7B_caption', 'qwen2vl7B_caption','LLaVA-Video-7B', "Qwen2.5-VL-72B", "gemini2.5_pre_flash", "gemini2.5_pro-05-06", "gpt4o_cap"
]

# 定义要统计的列名
# 
columns_to_map = { 
                   'qwen2vl7B_caption': "Qwen2VL-7B", 
                    'InternVL2.5-8B_caption': "InternVL2.5-8B", 
                    'VILA-8B_caption': "NVILA-8B", 
                    'LLaVA-Video-7B': 'LLaVA-Video-7B',
                    'Qwen2.5-VL-7B': "Qwen2.5-VL-7B", 
                    'VideoLLaMA3-7B_caption': "VideoLLaMA3-7B",
                    "Qwen2.5-VL-72B": "Qwen2.5-VL-72B",
                    # 'gemini_caption': "Gemini-1.5", 
                    "gemini2.5_pre_flash": "Gemini2.5-Pro-Flash",
                    "gemini2.5_pro-05-06": "Gemini-2.5-Pro-Preview",
                    "gpt4o_cap": "GPT-4o"
                    }


# 计算每列的单词长度
word_lengths_dict = {}
# 计算每列的平均单词长度
average_word_lengths = {}
for column in columns_to_analyze:
    # 计算每行的单词长度，忽略特殊符号
    # word_lengths = df[column].apply(lambda x: [len(word) for word in re.findall(r'\b\w+\b', str(x))])
    # 计算每行的单词长度，忽略特殊符号
    word_lengths = df[column].apply(lambda x: len(re.findall(r'\b\w+\b', str(x))))

    # 将所有单词长度展平为一个列表
    # word_lengths_flat = [length for sublist in word_lengths for length in sublist]
    word_lengths_dict[columns_to_map[column]] = word_lengths

    average_length = word_lengths.mean()
    average_word_lengths[column] = average_length

# 打印结果
for column, avg_length in average_word_lengths.items():
    print(f"Average word length in column '{column}': {avg_length:.2f}")


# 画直方图
plt.figure(figsize=(15, 10))

# 为每个列画一个直方图
for column, word_lengths in word_lengths_dict.items():
    plt.hist(word_lengths, bins=range(10, 800, 15), alpha=0.5, label=column, edgecolor='black')

plt.xlabel('Caption Word Length', fontproperties=prop)
plt.ylabel('Frequency', fontproperties=prop)

# 设置x轴和y轴刻度值的字体
plt.xticks(fontproperties=prop2)
plt.yticks(fontproperties=prop2)

# plt.title('Word Length Frequency Distribution in Each Caption Column')
plt.legend(loc='upper right', prop=prop2)
plt.tight_layout()

# 保存为无白边的PDF
plt.savefig('word_length_distribution-all-3.pdf', format='pdf', bbox_inches='tight', pad_inches=0)

# 显示图表
plt.show()