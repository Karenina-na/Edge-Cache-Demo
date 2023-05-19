# A3C参数
UPDATE_GLOBAL_ITER = 20
PARALLEL_NUM = 3
GAMMA = 0.99
MAX_EP = 10000
LEARNING_RATE = 1e-1
BETAS = (0.92, 0.999)
MODEL_PATH = None

# DQN
EPSILON_START = 0.5
EPSILON_END = 0.02
EPSILON_DECAY = 100000
TARGET_UPDATE_FREQUENCY = 20
n_episode = 1000
n_time_step = 1000
model_path = None

# 环境参数
S_dim = 15  # 文件个数
cache_space = 4  # 缓存空间大小
Stop_number = 200  # 仿真请求总数
Request_number = 30  # 一次请求个数
w = 1  # 流行度变化幅度
Zipf_alpha = 0.6  # zipf分布参数
Zipf_baseline = 0.062  # 低于某个阈值的内容不会被请求
Baseline = 50  # 奖励持续基线
