from Env.env import Env
import numpy as np
from Main.Env.param import *

env = Env(S_dim, A_dim, Request_number, Stop_number)
_, _ = env.reset()
lfu = [-1 for i in range(S_dim)]
for i in range(1000):
    obs, _, _, _, _ = env.step(lfu)
    obs = np.array(obs).reshape(3, A_dim)
    L = np.argsort(np.array(obs[0]))[-A_number:]
    lfu = np.zeros(shape=A_dim, dtype=int).tolist()
    for index in L:
        lfu[int(index)] = 1

print("cache hit ratio %f" % (env.cache / env.total))
print("cache time out %f" % (env.time_out_file / env.total))
