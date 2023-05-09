from Env.env import Env
from Main.Env.param import *
import numpy as np

# 需要将请求次数改为1
env = Env(S_dim, A_dim, 1, Stop_number * Request_number)
_, _ = env.reset()
fifo = [-1 for i in range(A_number)]
for i in range(Stop_number * Request_number):
    # 数据转换
    L = np.zeros(shape=A_dim, dtype=int).tolist()
    for data in fifo:
        if data != -1:
            L[data] = 1
    obs, _, _, _, _ = env.step(L)
    # 找到值为1的序号
    cache = 0
    for index in range(len(obs[0])):
        if obs[0][index] == 1:
            cache = index
            break
    fifo.append(cache)
    fifo.pop(0)

print("cache hit ratio %f" % (env.cache / env.total))
print("cache time out %f" % (env.time_out_file / env.total))
