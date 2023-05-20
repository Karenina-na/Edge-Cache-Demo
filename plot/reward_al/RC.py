from Env.env import Env
import numpy as np
from Param import *

env = Env()
_, _ = env.reset()
rc = [-1 for i in range(S_dim)]
cache_hit = 0
cache_total = 0
reward = []
for i in range(10000):
    a = np.random.choice(env.action_space.actions_index_number, 1, replace=False)
    s, r, done, info, _ = env.step(a[0])  # trunk,info will not be used
    cache_hit += info['cache_hit']
    cache_total += info['cache_total']
    reward.append(r)

print('-' * 100)
print('cache hit rate:', cache_hit/cache_total)

# 保存reward
reward = np.array(reward)
import pandas as pd
reward = pd.DataFrame(reward)
reward.to_csv('rc_reward.csv', index=False, header=False)