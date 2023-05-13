import torch
import torch.multiprocessing as mp
import os
from Agent.A3C_Agent import Agent, SharedAdam, ActionSpace
import numpy as np
from Main.Env.env import Env
from Main.Env.param import *

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
    def __init__(self, global_net: Agent, optimizer: torch.optim, g_ep, g_ep_r, result_queue, num: int, env,
                 action_space):
        super(Worker, self).__init__()
        self.name = 'w%02i' % num
        self.g_ep, self.g_ep_r, self.res_queue = g_ep, g_ep_r, result_queue
        self.gnet, self.opt = global_net, optimizer
        # 创建局部网络
        self.lnet = Agent(global_net.s_dim, global_net.a_dim, GAMMA, model_path=MODEL_PATH)
        self.env = env
        self.action_space = action_space

    def run(self):
        """
        运行
        :return: None
        """
        total_step = 1
        while self.g_ep.value < MAX_EP:
            state, _ = self.env.reset()
            # 一局游戏的数据
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            while True:
                # 动作索引
                a = self.lnet.choose_action(v_wrap(state[None, :]))
                a_index = self.action_space.dic[a]
                print(a_index)
                a_one_hot = np.zeros(A_dim, dtype=np.int32)
                for index in a_index:
                    a_one_hot[index] = 1
                next_state, reward, done, _, _ = self.env.step(a_one_hot)
                # 游戏结束，给予惩罚
                if done:
                    reward = -1
                ep_r += reward
                buffer_a.append(a)
                buffer_s.append(state)
                buffer_r.append(reward)
                # 更新全局网络
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    push_and_pull(self.opt, self.lnet, self.gnet, done, buffer_s, buffer_a, buffer_r, next_state)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    # 若游戏结束，退出循环
                    if done:
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                state = next_state
                total_step += 1
        self.res_queue.put(None)
        print(self.name, "is close!")


def v_wrap(np_array, dtype=np.float32):
    """Convert numpy array to torch tensor"""
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def push_and_pull(optimizer: torch.optim, local_net: Agent, global_net: Agent, done, bs, ba, br, next_state):
    """
    更新全局网络
    :param optimizer: 优化器
    :param local_net: 局部网络
    :param global_net: 全局网络
    :param done: 游戏是否结束
    :param bs: 一个batch存储的状态
    :param ba: 一个batch存储的动作
    :param br: 一个batch存储的奖励
    :param next_state: 下一个状态
    :return:
    """
    if done:
        value_next_state = 0.  # terminal
    else:
        value_next_state = local_net.forward(v_wrap(next_state[None, :]))[-1].data.numpy()[0, 0]

    # 计算目标累计奖励 target = r + gamma * V(s')
    buffer_v_target = []
    for reward in br[::-1]:
        value_next_state = reward + GAMMA * value_next_state
        buffer_v_target.append(value_next_state)
    buffer_v_target.reverse()

    # 计算损失函数
    actor_loss, critic_loss = \
        local_net.loss_func(
            v_wrap(np.vstack(bs)),
            v_wrap(np.array(ba), dtype=np.int64),
            v_wrap(np.array(buffer_v_target)[:, None]))

    # 反向传播，和全局网络同步
    loss = actor_loss + critic_loss
    optimizer.zero_grad()
    loss.backward()
    # critic_loss.backward(retain_graph=True)  # 保留计算图
    # actor_loss.backward()
    for lp, gp in zip(local_net.parameters(), global_net.parameters()):
        gp._grad = lp.grad * TAU
    optimizer.step()

    # 同步局部网络
    local_net.load_state_dict(global_net.state_dict())


UPDATE_GLOBAL_ITER = 100
PARALLEL_NUM = 6
GAMMA = 0.9
MAX_EP = 1000
LEARNING_RATE = 1e-1
BETAS = (0.92, 0.999)
TAU = 0.1
MODEL_PATH = "../Result/checkpoints"


def train():
    env = Env(S_dim, A_dim, Request_number, Stop_number)
    action_space = ActionSpace(A_number, A_dim)
    N_S = S_dim
    N_A = action_space.n_action
    gnet = Agent(3 * N_S, N_A, GAMMA, MODEL_PATH)  # global network

    gnet.share_memory()  # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=LEARNING_RATE, betas=BETAS)  # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # 并发训练
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i, env, action_space) for i in range(PARALLEL_NUM)]
    print("Start training...")

    [w.start() for w in workers]
    res = []
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    res = np.array(res)
    [w.join() for w in workers]

    print("Training finished!")
    import matplotlib.pyplot as plt

    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    # plt.show()

    # 玩游戏
    env_test = Env(S_dim, A_dim, Request_number, Stop_number)
    action_space = ActionSpace(A_number, A_dim)
    s, _ = env_test.reset()
    gnet.type = 'test'
    while True:
        a = gnet.choose_action(v_wrap(s[None, :]))
        a_index = action_space.dic[a]
        a_one_hot = np.zeros(A_dim, dtype=np.int32)
        for index in a_index:
            a_one_hot[index] = 1
        s, _, d, _, _ = env_test.step(a_one_hot)
        if d:
            break

    print("last network cache hit ratio %f" % (env_test.cache / env_test.total))
    print("last network cache time out %f" % (env_test.time_out_file / env_test.total))

    # 保存模型
    gnet.save_model()


def test():
    env_test = Env(S_dim, A_dim, Request_number, Stop_number)
    action_space = ActionSpace(A_number, A_dim)
    N_S = S_dim
    N_A = action_space.n_action
    gnet = Agent(3 * N_S, N_A, GAMMA, MODEL_PATH)  # global network
    s, _ = env_test.reset()
    gnet.type = "test"
    while True:
        # 动作索引
        a = gnet.choose_action(v_wrap(s[None, :]))
        a_index = action_space.dic[a]
        a_one_hot = np.zeros(A_dim, dtype=np.int32)
        for index in a_index:
            a_one_hot[index] = 1
        # 步入
        s, _, d, _, _ = env_test.step(a_one_hot)
        if d:
            break

    print("cache hit ratio %f" % (env_test.cache / env_test.total))
    print("last network cache time out %f" % (env_test.time_out_file / env_test.total))


if __name__ == "__main__":
    train()
    # test()
