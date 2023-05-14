from Env.env import Env
import numpy as np
from Main.Env.param import *

env = Env()
_, _ = env.reset()
rc = [-1 for i in range(S_dim)]
cache_hit = [0, 0, 0]
cache_total = [0, 0, 0]
node_time_out = [0, 0, 0]
for i in range(10000):
    a = np.random.choice(env.action_space.actions_index_number, A_number, replace=False)
    s, r, done, info, _ = env.step(a)  # trunk,info will not be used
    s = np.swapaxes(s, 0, 1)
    s = np.reshape(s, newshape=(len(s), -1))
    print(s[1])
    cache_hit += np.reshape(info["cache_hit"], newshape=(Node_number,))
    cache_total += np.reshape(info["cache_total"], newshape=(Node_number,))
    node_time_out += np.reshape(info["node_timeout"], newshape=(Node_number,))

print('-'*100)
for i in range(Node_number):
    print("node %d cache hit ratio %f" % (i, cache_hit[i] / cache_total[i]))
print()
for i in range(Node_number):
    print("node %d cache time out %f" % (i, node_time_out[i] / cache_total[i]))



