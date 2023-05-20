# DQN
EPSILON_START = 0.8
EPSILON_END = 0.02
EPSILON_DECAY = 100000
TARGET_UPDATE_FREQUENCY = 20
n_episode = 1000
n_time_step = 1000
model_path = None

# 环境参数
S_dim = 15  # 文件个数 15
cache_space = 4  # 缓存空间大小 4
Stop_number = 200  # 仿真请求总数 200
Request_number = 30  # 一次请求个数起始 30
Request_number_max = 50  # 一次请求个数上限 50
Zipf_w = 1  # 流行度变化幅度 1
Zipf_step = 2  # zipf分布变化步长 2
Zipf_alpha = 0.6  # zipf分布参数 0.6
Zipf_baseline = 0.062  # 低于某个阈值的内容不会被请求 0.062
Baseline = 50  # 奖励持续基线 50
file_w = [0.1, 0.9, 0.5] # 0.1, 0.9, 0.5
file_scale = [20000, 70000]  # 文件大小缩放范围 20000, 70000
