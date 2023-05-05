import torch
import torch.multiprocessing as mp
import os
from Agent.A3C_Agent import Agent, SharedAdam
import numpy as np
from Env.env import Env

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def record(g_ep, g_ep_r, ep_r, result_queue, name):
    """
    记录每一局游戏的结果
    :param g_ep: 总局数
    :param g_ep_r: 总奖励
    :param ep_r: 本局奖励
    :param result_queue: 结果队列
    :param name: 进程名
    :return:
    """
    with g_ep.get_lock():
        g_ep.value += 1
    with g_ep_r.get_lock():
        if g_ep_r.value == 0.:
            g_ep_r.value = ep_r
        else:
            g_ep_r.value = g_ep_r.value * 0.99 + ep_r * 0.01
    result_queue.put(g_ep_r.value)
    print(name, "Ep:", g_ep.value, "| Ep_r: %.0f" % g_ep_r.value)


class Worker(mp.Process):
    def __init__(self, global_net: Agent, optimizer: torch.optim, g_ep, g_ep_r, result_queue, num: int, env):
        super(Worker, self).__init__()
        self.name = 'w%02i' % num
        self.g_ep, self.g_ep_r, self.res_queue = g_ep, g_ep_r, result_queue
        self.gnet, self.opt = global_net, optimizer
        # 创建局部网络
        self.lnet = Agent(global_net.s_dim, global_net.a_dim, GAMMA, model_path=MODEL_PATH,
                          a_number=global_net.a_number)
        self.env = env

    def run(self):
        """
        运行
        :return: None
        """
        total_step = 1
        while self.g_ep.value < MAX_EP:
            state, _ = self.env.reset()
            # 一局游戏的数据
            buffer_s, buffer_a, buffer_r, buffer_n_s = [], [], [], []
            ep_r = 0.
            while True:
                # 选择动作并执行
                action = self.lnet.choose_action(v_wrap(state[None, :]))
                next_state, reward, done, _, _ = self.env.step(action[0])
                ep_r += reward
                buffer_a.append(action[0])
                buffer_s.append(state)
                buffer_r.append(reward)
                buffer_n_s.append(next_state)
                # 更新全局网络
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    push_and_pull(self.opt, self.lnet, self.gnet, done, buffer_s, buffer_a, buffer_r, buffer_n_s)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    # 若游戏结束，退出循环
                    if done:
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                state = next_state
                total_step += 1
        self.res_queue.put(None)


def v_wrap(np_array, dtype=np.float32):
    """Convert numpy array to torch tensor"""
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def push_and_pull(optimizer: torch.optim, local_net: Agent, global_net: Agent, done, bs, ba, br, n_bs):
    """
    更新全局网络
    :param optimizer: 优化器
    :param local_net: 局部网络
    :param global_net: 全局网络
    :param done: 游戏是否结束
    :param bs: 一个batch存储的状态
    :param ba: 一个batch存储的动作
    :param br: 一个batch存储的奖励
    :param n_bs: 一个batch存储的下一个状态
    :return:
    """
    # 计算损失函数
    ba = np.array(ba)
    actor_loss, critic_loss = \
        local_net.loss_func(
            v_wrap(np.array(bs)),
            v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
            v_wrap(np.array(br)),
            v_wrap(np.array(n_bs)),
            GAMMA
        )

    # loss = actor_loss + critic_loss
    loss = actor_loss + critic_loss

    # 反向传播，和全局网络同步
    optimizer.zero_grad()
    loss.backward()
    for lp, gp in zip(local_net.parameters(), global_net.parameters()):
        gp._grad = lp.grad
    optimizer.step()

    # 同步局部网络
    local_net.load_state_dict(global_net.state_dict())


UPDATE_GLOBAL_ITER = 10
PARALLEL_NUM = 14
GAMMA = 0.9
MAX_EP = 1000
LEARNING_RATE = 1e-2
BETAS = (0.92, 0.999)
MODEL_PATH = "../Result/checkpoints"
# MODEL_PATH = None
A_dim = 20  # 缓存内容索引大小
S_dim = 20  # 状态空间
A_number = 4  # 缓存空间大小
Request_number = 100  # 一次请求的请求数量
Stop_number = 10000  # 环境请求最大数量


def train():
    env = Env(S_dim, A_dim, Request_number, Stop_number)
    N_S = S_dim
    N_A = A_dim

    gnet = Agent(3 * N_S, N_A, GAMMA, MODEL_PATH, A_number)  # global network
    gnet.share_memory()  # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=LEARNING_RATE, betas=BETAS)  # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # 初始
    env_test = Env(S_dim, A_dim, Request_number, Stop_number)
    s, _ = env_test.reset()
    while True:
        a = gnet.choose_action(v_wrap(s[None, :]))
        s, _, d, _, _ = env_test.step(a[0])
        if d:
            break
    env_test.close()

    # 并发训练
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i, env) for i in range(PARALLEL_NUM)]
    [w.start() for w in workers]

    res = []  # 记录每一步的平均奖励
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    res = np.array(res)
    [w.join() for w in workers]

    env.close()

    import matplotlib.pyplot as plt

    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()

    print("game over")
    print("Init network cache hit ratio %f" % (env_test.cache / env_test.total))
    print("Init network cache time out %f" % (env_test.time_out_file / (env_test.total - env_test.cache)))
    # 玩游戏
    env_test = Env(S_dim, A_dim, Request_number, Stop_number)
    s, _ = env_test.reset()
    while True:
        a = gnet.choose_action(v_wrap(s[None, :]))
        s, _, d, _, _ = env_test.step(a[0])
        if d:
            break
    env_test.close()

    print("last network cache hit ratio %f" % (env_test.cache / env_test.total))
    print("last network cache time out %f" % (env_test.time_out_file / (env_test.total - env_test.cache)))
    # 保存模型
    gnet.save_model()


def test():
    env_test = Env(S_dim, A_dim,  Request_number, Stop_number)
    N_S = S_dim
    N_A = A_dim
    gnet = Agent(3 * N_S, N_A, GAMMA, MODEL_PATH, A_number)  # global network
    s, _ = env_test.reset()
    while True:
        a = gnet.choose_action(v_wrap(s[None, :]))
        s, _, d, _, _ = env_test.step(a[0])
        if d:
            break
    env_test.close()

    print("cache hit ratio %f" % (env_test.cache / env_test.total))


if __name__ == "__main__":
    train()
    # test()
