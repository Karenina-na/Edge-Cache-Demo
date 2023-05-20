from Env.env import Env
import numpy as np
from Param import *

env = Env()
_, _ = env.reset()
time_out = 0
last_request = []
for i in range(10000):
    a = np.random.choice(env.action_space.actions_index_number, 1, replace=False)
    # 更新上一时刻的请求
    request = last_request
    last_request = env.request
    # 统计频率
    dic = {}
    for req in request:
        if req in dic.keys():
            dic[req] += 1
        else:
            dic[req] = 1
    # 按频率排序
    dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    # 选取频率最高的cache_space个
    cache_index = []
    for i in range(len(dic)):
        cache_index.append(dic[i][0])
        if len(cache_index) == cache_space:
            break
    # 找到对应的action
    a = np.random.choice(env.action_space.actions_index_number, 1, replace=False)[0]
    action_dic = env.action_space.action_index_dic
    for i in range(len(action_dic)):
        if action_dic[i] == cache_index:
            a = i
            break
    s, r, done, info, _ = env.step(a)  # trunk,info will not be used
    time_out += info['time_out']

print('-' * 100)
print('time out rate:', time_out / 10000)
