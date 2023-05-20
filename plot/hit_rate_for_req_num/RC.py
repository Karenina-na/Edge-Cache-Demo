from Env.env import Env
import numpy as np
from Param import *

env = Env()
_, _ = env.reset()
rc = [-1 for i in range(S_dim)]
cache_hit = []
cache_total = []
for i in range(10000):
    a = np.random.choice(env.action_space.actions_index_number, 1, replace=False)
    s, r, done, info, _ = env.step(a[0])  # trunk,info will not be used
    cache_hit.append(info['cache_hit'])
    cache_total.append(info['cache_total'])

print('-' * 100)
print('cache hit rate:', sum(cache_hit) / sum(cache_total))
