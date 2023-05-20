from Env.env import Env
import numpy as np
from Param import *

env = Env()
_, _ = env.reset()
cache_hit = 0
cache_total = 0
last_request = []
for i in range(10000):
    a = np.random.choice(env.action_space.actions_index_number, 1, replace=False)
    # 更新上一时刻的请求
    request = last_request
    last_request = env.request.copy()
    # 先进先出
    count = 0
    cache_index=[]
    for req in request:
        if req != -1 and req not in cache_index:
            cache_index.append(req)
            count += 1
            if count == cache_space:
                break

    # 找到对应的action
    a = np.random.choice(env.action_space.actions_index_number, 1, replace=False)[0]
    action_dic = env.action_space.action_index_dic
    for i in range(len(action_dic)):
        if np.array_equal(action_dic[i], cache_index):
            a = i
            break
    s, r, done, info, _ = env.step(a)  # trunk,info will not be used
    cache_hit += info['cache_hit']
    cache_total += info['cache_total']

print('-' * 100)
print('cache hit rate:', cache_hit / cache_total)
