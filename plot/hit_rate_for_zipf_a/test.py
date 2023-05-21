# 环境参数
S_dim = 15  # 文件个数
cache_space = 4  # 缓存空间大小
Stop_number = 200  # 仿真请求总数
Request_number = 30  # 一次请求个数起始
Request_number_max = 50  # 一次请求个数上限
Zipf_w = 1  # 流行度变化幅度
Zipf_step = 2  # zipf分布变化步长
Zipf_baseline = 0.062  # 低于某个阈值的内容不会被请求
Baseline = 50  # 奖励持续基线

Zipf_alpha = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
fifo = [0.473, 0.480, 0.520, 0.532, 0.541, 0.547, 0.557, 0.568, 0.589]
lfu = [0.477, 0.486, 0.560, 0.577, 0.592, 0.600, 0.607, 0.617, 0.629]
rc = [0.268, 0.267, 0.270, 0.263, 0.269, 0.269, 0.258, 0.266, 0.267]
dqn = [0.762, 0.767, 0.779, 0.820, 0.842, 0.855, 0.877, 0.889, 0.891]

# 放大一百倍
Zipf_alpha = [int(i * 100) for i in Zipf_alpha]
fifo = [i * 100 for i in fifo]
lfu = [i * 100 for i in lfu]
rc = [i * 100 for i in rc]
dqn = [i * 100 for i in dqn]

import numpy as np
import pandas as pd
from doc.Plot.require import *

request_num = pd.DataFrame(Zipf_alpha)
fifo = pd.DataFrame(fifo)
lfu = pd.DataFrame(lfu)
rc = pd.DataFrame(rc)
dqn = pd.DataFrame(dqn)

# Import Data
df = pd.concat([request_num, fifo, lfu, rc, dqn], axis=1)
df.columns = ['step', 'FIFO', 'LFU', 'RC', 'DQN']

print(df.head())

# Define the upper limit, lower limit, interval of Y axis and colors
y_LL = 10
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
    plt.hlines(y, xmin=40, xmax=request_num.iloc[-1].values[0] - 10, colors='black', alpha=0.3, linestyles="--", lw=0.5)

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

print(request_num.iloc[0].values[0], request_num.iloc[-1].values[0])
plt.xticks(range(request_num.iloc[0].values[0] - 10, request_num.iloc[-1].values[0] - 5, 5), df['step'].values[::1], horizontalalignment='left',
           fontsize=12)
plt.ylim(y_LL, y_UL)

plt.xlim(39, request_num.iloc[-1].values[0] - 9)
plt.ylabel('The CHR of different algorithms', fontsize=14)
plt.xlabel('The number of requests', fontsize=14)

plt.legend(loc='lower right', ncol=1, fontsize=12)
plt.savefig("../../Result/images/Zipf_A.png", dpi=300)
plt.show()
