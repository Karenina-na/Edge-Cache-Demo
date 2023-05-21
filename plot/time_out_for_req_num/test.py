# 环境参数
S_dim = 15  # 文件个数
cache_space = 4  # 缓存空间大小
Stop_number = 200  # 仿真请求总数
Zipf_w = 1  # 流行度变化幅度
Zipf_step = 2  # zipf分布变化步长
Zipf_alpha = 0.6  # zipf分布参数
Zipf_baseline = 0.062  # 低于某个阈值的内容不会被请求
Baseline = 50  # 奖励持续基线
file_w = [0.1, 0.9, 0.5]
file_scale = [20000, 70000]  # 文件大小缩放范围

# fifo lfu rc dqn
request_num = [10, 20, 30, 40, 50, 60, 70, 80]
fifo = [4.42, 77.79, 162.20, 249.30, 338.09, 416.34, 504.37, 593.51]
lfu = [4.20, 97.07, 163.71, 240.62, 326.33, 397.65, 481.83, 567.74]
rc = [4.35, 100.50, 204.52, 310.72, 419.16, 520.22, 624.60, 732.49]
dqn = [4.59, 50.35, 99.8, 162.16, 198.84, 218.63, 261.70, 320.28]

import numpy as np
import pandas as pd
from doc.Plot.require import *

request_num = pd.DataFrame(request_num)
fifo = pd.DataFrame(fifo)
lfu = pd.DataFrame(lfu)
rc = pd.DataFrame(rc)
dqn = pd.DataFrame(dqn)

# Import Data
df = pd.concat([request_num, fifo, lfu, rc, dqn], axis=1)
df.columns = ['step', 'FIFO',  'LFU', 'RC', 'DQN']
print(df.head())
# Define the upper limit, lower limit, interval of Y axis and colors
y_LL = 0
y_UL = 800
y_interval = 100
my_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown']
my_line_style = ['-', '--', '-.', ':', '-']
my_marker_style = ['o', '*', 's', 'D', 'P']

# Draw Plot and Annotate
fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=300)

columns = df.columns[1:]
for i, column in enumerate(columns):
    plt.plot(df['step'].values - 10, df[column].values, lw=2, color=my_colors[i], label=column,
             linestyle=my_line_style[i], marker=my_marker_style[i], markersize=6)
    # plt.text(request_num.iloc[-1].values[0] - 8, df[column].values[-1], column, fontsize=14, color=my_colors[i])

# Draw Tick lines
for y in range(y_LL, y_UL, y_interval):
    plt.hlines(y, xmin=0, xmax=request_num.iloc[-1].values[0] - 10, colors='black', alpha=0.3, linestyles="--", lw=0.5)

# Draw Tick lines
for x in range(0, request_num.iloc[-1].values[0], 10):
    plt.axvline(x=x, color='black', alpha=0.3, linestyle="--", linewidth=0.5)

# Decorations
plt.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=True, left=False, right=False,
                labelleft=True)

# Lighten borders
plt.gca().spines["top"].set_alpha(0.3)
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)
plt.gca().spines["left"].set_alpha(0.3)

plt.yticks(range(y_LL, y_UL, y_interval), [str(y) for y in range(y_LL, y_UL, y_interval)], fontsize=12)
plt.xticks(range(0, request_num.iloc[-1].values[0], 10), df['step'].values[::1], horizontalalignment='left', fontsize=12)
plt.ylim(y_LL, y_UL)
plt.xlim(-2, request_num.iloc[-1].values[0] - 5)
plt.ylabel('The time latency of different algorithms (ms)', fontsize=14)
plt.xlabel('The number of requests', fontsize=14)

plt.legend(loc='lower right', ncol=1, fontsize=12)
plt.show()
