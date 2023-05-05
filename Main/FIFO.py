from Env.env import Env


A_dim = 20  # 缓存内容索引大小
S_dim = 20  # 状态空间
A_number = 4  # 缓存空间大小
Request_number = 100  # 一次请求的请求数量
Stop_number = 10000  # 环境请求最大数量
env = Env(S_dim, A_dim, Request_number, Stop_number)
_, _ = env.reset()
fifo = [-1 for i in range(S_dim)]
for i in range(1000):
    obs, _, _, _, _ = env.step(fifo)
