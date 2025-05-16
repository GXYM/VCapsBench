import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# Path to the font file
font_path = 'times.ttf'
# Create a font object with bold weight and specific size
prop = fm.FontProperties(fname=font_path, weight='bold', size=28)
prop1 = fm.FontProperties(fname=font_path, weight='bold', size=22)
prop2 = fm.FontProperties(fname=font_path, weight='bold', size=25)
prop3 = fm.FontProperties(fname=font_path, weight='bold', size=12)
prop4 = fm.FontProperties(fname=font_path, weight='bold', size=14)


datas1 = {
"Gemini-2.5-Pro-Preview": 534.48,
"Gemini2.5-Pro-Flash": 610.12,
"GPT-4o": 453.43,
'Qwen2.5-VL-72B': 318.77, 
'Qwen2VL-7B': 149.83, 
'InternVL2.5-8B': 374.64, 
'NVILA-8B': 375.39, 
'LLaVA-Video-7B': 165.85, 
'Qwen2.5-VL-7B': 321.03, 
'VideoLLaMA3-7B': 160.98
}

datas2 = {"Gemini-2.5-Pro-Preview": 85.52, "Gemini2.5-Pro-Flash": 84.42, "GPT-4o": 74.07,
'Qwen2.5-VL-72B': 70.86, 'Qwen2VL-7B': 52.52, 'InternVL2.5-8B': 63.34, 'NVILA-8B': 57.21, 'LLaVA-Video-7B': 53.08, 'Qwen2.5-VL-7B': 65.72, 'VideoLLaMA3-7B': 54.13}

datas3 = {"Gemini-2.5-Pro-Preview": 10.28, "Gemini2.5-Pro-Flash": 10.49, "GPT-4o": 10.24, 'Qwen2.5-VL-72B': 10.98,'Qwen2VL-7B': 9.65, 'InternVL2.5-8B': 13.41, 'NVILA-8B': 13.12, 'LLaVA-Video-7B': 10.05, 'Qwen2.5-VL-7B': 10.62, 'VideoLLaMA3-7B': 10.50}


# Sort the data based on datas1
sorted_keys = sorted(datas1, key=datas1.get, reverse=False)
sorted_datas1 = {k: datas1[k] for k in sorted_keys}
sorted_datas2 = {k: datas2[k] for k in sorted_keys}
sorted_datas3 = {k: datas3[k] for k in sorted_keys}

# Extract sorted keys and values
labels = list(sorted_datas1.keys())
values1 = list(sorted_datas1.values())
values2 = [d*2.5 for d in list(sorted_datas2.values())]
values3 = [d*2.5 for d in list(sorted_datas3.values())]

# Set the position and width of the bars
x = np.arange(len(labels))
width = 0.3  # Adjusted width for three datasets

# Create the figure and axes
fig, ax = plt.subplots(figsize=(18, 9))

# Draw the bar charts
bars1 = ax.bar(x - width, values1, width, label='Caption Length')
bars2 = ax.bar(x, values2, width, label='Coverage Rate (CR)')
bars3 = ax.bar(x + width, values3, width, label='Inconsistency Rate (IR)')

# Add labels and title
ax.set_xlabel('VLMs', fontproperties=prop)
ax.set_ylabel('Values', fontproperties=prop)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=22.5, ha='right', fontproperties=prop2)

ax.yaxis.set_ticks([])

ax.legend(prop=prop1)

# Display value labels on the bars
def add_value_labels(bars, rate_flag=False):
    for bar in bars:
        height = bar.get_height()
        if rate_flag:
            ax.annotate(f'{(height/2.5):.2f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontproperties=prop3)
        else:
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontproperties=prop4)


add_value_labels(bars1)
add_value_labels(bars2, rate_flag=True)
add_value_labels(bars3, rate_flag=True)

# Adjust layout to fit labels
plt.tight_layout()

# Save the figure as a PDF file, removing white borders
plt.savefig('comparison_chart_sorted_NEW2.pdf', format='pdf', bbox_inches='tight')

# Show the figure
plt.show()