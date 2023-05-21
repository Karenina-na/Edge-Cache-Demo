# 环境参数
S_dim = 15  # 文件个数 15
cache_space = 4  # 缓存空间大小 4
Stop_number = 200  # 仿真请求总数 200
Zipf_w = 1  # 流行度变化幅度 1
Zipf_step = 2  # zipf分布变化步长 2
Zipf_alpha = 0.6  # zipf分布参数 0.6
Zipf_baseline = 0.062  # 低于某个阈值的内容不会被请求 0.062
Baseline = 50  # 奖励持续基线 50
file_w = [0.1, 0.9, 0.5]  # 0.1, 0.9, 0.5
file_scale = [20000, 70000]  # 文件大小缩放范围 20000, 70000

# fifo lfu rc dqn
request_num = [10, 20, 30, 40, 50, 60, 70, 80]
fifo = [0.27, 0.45, 0.520, 0.532, 0.521, 0.51, 0.50, 0.49]
lfu = [0.25, 0.32, 0.42, 0.56, 0.55, 0.57, 0.53, 0.52]
rc = [0.26, 0.27, 0.27, 0.268, 0.270, 0.272, 0.267, 0.261]
dqn = [0.31, 0.51, 0.65, 0.73, 0.83, 0.80, 0.77, 0.78]

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
df.columns = ['step', 'FIFO', 'LFU', 'RC', 'DQN']

# Define the upper limit, lower limit, interval of Y axis and colors
y_LL = 9
y_UL = 91
y_interval = 10
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
    plt.axvline(x=x, ymin=0.01, ymax=0.99, color='black', alpha=0.3, linestyle="--", linewidth=0.5)

# Decorations
plt.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=True, left=False, right=False,
                labelleft=True)

# Lighten borders
plt.gca().spines["top"].set_alpha(0.3)
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)
plt.gca().spines["left"].set_alpha(0.3)

plt.yticks(range(y_LL, y_UL, y_interval), [str(y / 100) for y in range(y_LL, y_UL, y_interval)], fontsize=12)

plt.xticks(range(0, request_num.iloc[-1].values[0], 10), df['step'].values[::1], horizontalalignment='left',
           fontsize=12)
plt.ylim(y_LL, y_UL)
plt.xlim(-2, request_num.iloc[-1].values[0] - 7)
plt.ylabel('The CHR of different algorithms', fontsize=14)
plt.xlabel('The number of requests', fontsize=14)

plt.legend(loc='lower right', ncol=1, fontsize=12)
plt.savefig("../../Result/images/CHR.png", dpi=300)
plt.show()
