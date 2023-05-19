from Env.env import Env
import numpy as np
from param import *

env = Env()
_, _ = env.reset()
rc = [-1 for i in range(S_dim)]
cache_hit = 0
cache_total = 0
for i in range(10000):
    a = np.random.choice(env.action_space.actions_index_number, 1, replace=False)
    s, r, done, info, _ = env.step(a[0])  # trunk,info will not be used
    cache_hit += info['cache_hit']
    cache_total += info['cache_total']

print('-' * 100)
print('cache hit rate:', cache_hit/cache_total)
