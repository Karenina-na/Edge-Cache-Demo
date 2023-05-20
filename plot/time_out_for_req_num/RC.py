from Env.env import Env
import numpy as np
from Param import *

env = Env()
_, _ = env.reset()
rc = [-1 for i in range(S_dim)]
time_out = 0
for i in range(10000):
    a = np.random.choice(env.action_space.actions_index_number, 1, replace=False)
    s, r, done, info, _ = env.step(a[0])  # trunk,info will not be used
    time_out += info['time_out']

print('-' * 100)
print('time out rate:', time_out / 10000)
