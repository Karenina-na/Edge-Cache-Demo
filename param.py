# A3C参数
UPDATE_GLOBAL_ITER = 10
PARALLEL_NUM = 1
GAMMA = 0.9
MAX_EP = 1000
LEARNING_RATE = 1e-1
BETAS = (0.92, 0.999)
MODEL_PATH = "Result/checkpoints"

# DQN
EPSILON_START = 0.1
EPSILON_END = 0.02
EPSILON_DECAY = 100000
TARGET_UPDATE_FREQUENCY = 20
n_episode = 100
n_time_step = 1000
model_path = ""

# 环境参数
S_dim = 10  # 文件个数
cache_space = 3  # 缓存空间大小
Stop_number = 20000  # 仿真请求总数
Request_number = 20  # 一次请求个数
w = 10000  # 流行度变化幅度
Zipf_alpha = 1  # zipf分布参数
