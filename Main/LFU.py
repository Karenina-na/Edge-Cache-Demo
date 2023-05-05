from Env.env import Env
import numpy as np

A_dim = 20  # 缓存内容索引大小
S_dim = 20  # 状态空间
A_number = 4  # 缓存空间大小
Request_number = 100  # 一次请求的请求数量
Stop_number = 10000  # 环境请求最大数量
env = Env(S_dim, A_dim, Request_number, Stop_number)
_, _ = env.reset()
LFU = [-1 for i in range(S_dim)]
for i in range(1000):
    obs, _, _, _, _ = env.step(LFU)
    List = np.argsort(np.array(obs[0]))[:-3]
    LFU = np.zeros(shape=A_dim, dtype=int).tolist()
    for index in List:
        LFU[int(index)] = 1

print("cache hit ratio %f" % (env.cache / env.total))
print("cache time out %f" % (env.time_out_file / (env.total - env.cache)))
