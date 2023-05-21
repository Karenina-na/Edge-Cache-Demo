import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from doc.Plot.require import *

from scipy.signal import savgol_filter


def savitzky_golay(data, window_size, order):
    """
    数据平滑
    """
    smoothed_data = savgol_filter(data, window_size, order)
    return smoothed_data


# Import Data
reward = pd.read_csv('reward.csv', header=None)
fifo_reward = pd.read_csv('fifo_reward.csv', header=None)
lfu_reward = pd.read_csv('lfu_reward.csv', header=None)
rc_reward = pd.read_csv('rc_reward.csv', header=None)

# 相邻两个数据平均
# for i in range(0, reward.shape[0] - 1):
#     if i % 2 == 0:
#         reward.iloc[i] = (reward.iloc[i] + reward.iloc[i + 1]) / 2
#         reward.iloc[i + 1] = 0
#         fifo_reward.iloc[i] = (fifo_reward.iloc[i] + fifo_reward.iloc[i + 1]) / 2
#         fifo_reward.iloc[i + 1] = 0
#         lfu_reward.iloc[i] = (lfu_reward.iloc[i] + lfu_reward.iloc[i + 1]) / 2
#         lfu_reward.iloc[i + 1] = 0
#         rc_reward.iloc[i] = (rc_reward.iloc[i] + rc_reward.iloc[i + 1]) / 2
#         rc_reward.iloc[i + 1] = 0

# 平滑数据
reward = savitzky_golay(reward.iloc[:, 0].tolist(), 3, 2)
fifo_reward = savitzky_golay(fifo_reward.iloc[:, 0].tolist(), 70, 7)
lfu_reward = savitzky_golay(lfu_reward.iloc[:, 0].tolist(), 70, 7)
rc_reward = savitzky_golay(rc_reward.iloc[:, 0].tolist(), 100, 7)

reward = pd.DataFrame(reward)
fifo_reward = pd.DataFrame(fifo_reward)
lfu_reward = pd.DataFrame(lfu_reward)
rc_reward = pd.DataFrame(rc_reward)

# 截取reward的前50个数据
reward = reward.iloc[0:49, :]
fifo_reward = fifo_reward.iloc[0:49, :]
lfu_reward = lfu_reward.iloc[0:49, :]
rc_reward = rc_reward.iloc[0:49, :]

# 删除为0的行
# reward = reward[reward.iloc[:, 0] != 0]
# fifo_reward = fifo_reward[fifo_reward.iloc[:, 0] != 0]
# lfu_reward = lfu_reward[lfu_reward.iloc[:, 0] != 0]
# rc_reward = rc_reward[rc_reward.iloc[:, 0] != 0]

# 重置索引
reward = reward.reset_index(drop=True)
fifo_reward = fifo_reward.reset_index(drop=True)
lfu_reward = lfu_reward.reset_index(drop=True)
rc_reward = rc_reward.reset_index(drop=True)

# 生成x轴数据
x = pd.DataFrame(np.arange(0, reward.shape[0], 1))
print(x.shape)
print(reward.shape)
print(fifo_reward.shape)
print(lfu_reward.shape)
print(rc_reward.shape)

# Import Data
df = pd.concat([x, fifo_reward, lfu_reward, rc_reward, reward], axis=1)
df.columns = ['step',  'FIFO', 'LFU', 'RC', 'DQN']
my_line_style = ['-', '--', '-.', ':', '-']
my_marker_style = ['o', '*', 's', 'D', 'P']

print(df.head())
# Define the upper limit, lower limit, interval of Y axis and colors
y_LL = -35
y_UL = 1
y_interval = 5
my_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown']

# Draw Plot and Annotate
fig, ax = plt.subplots(1, 1, figsize=(16, 9), dpi=200)

columns = df.columns[1:]
for i, column in enumerate(columns):
    plt.plot(df['step'].values, df[column].values, lw=2, color=my_colors[i], label=column,
             linestyle=my_line_style[i], marker=my_marker_style[i], markersize=6)
    # plt.text(df.shape[0] + 1, df[column].values[-1], column, fontsize=14, color=my_colors[i])

# Draw Tick lines
for y in range(y_LL, y_UL, y_interval):
    plt.hlines(y, xmin=0, xmax=reward.shape[0] + 1, colors='black', alpha=0.3, linestyles="--", lw=0.5)

# Draw Tick lines
for x in range(0, x.iloc[-1].values[0] + 10, 10):
    plt.axvline(x=x, ymin=0, ymax=0.97, color='black', alpha=0.3, linestyle="--", linewidth=0.5)

# Decorations
plt.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=True, left=False, right=False,
                labelleft=True)

# Lighten borders
plt.gca().spines["top"].set_alpha(0.3)
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)
plt.gca().spines["left"].set_alpha(0.3)

plt.yticks(range(y_LL, y_UL, y_interval), [str(y) for y in range(y_LL, y_UL, y_interval)], fontsize=12)
plt.xticks(range(0, df.shape[0], 5), df['step'].values[::5] * 100, horizontalalignment='left', fontsize=12)
plt.ylim(y_LL, y_UL)
plt.xlim(-1, reward.shape[0] + 2)
plt.ylabel('Reward', fontsize=14)
plt.xlabel('Step', fontsize=14)

plt.legend(loc='lower right', ncol=1, fontsize=12)
plt.savefig("../../Result/images/Reward.png", dpi=300)
plt.show()
