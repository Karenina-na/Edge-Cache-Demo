import numpy as np
import torch
from Env.env import Env
from Agent.DDPG_Agent import ReplayBuffer, Agent


def train():
    # -------------------------------------- #
    # 模型构建
    # -------------------------------------- #
    # 经验回放池实例化
    replay_buffer = ReplayBuffer(capacity=buffer_size)
    # 模型实例化
    agent = Agent(n_states=n_states, n_hidden=n_hidden, n_actions=n_actions, action_bound=action_bound,
                  sigma=sigma, actor_lr=actor_lr, critic_lr=critic_lr, tau=tau, gamma=gamma, device=device, path=path)

    # -------------------------------------- #
    # 模型训练
    # -------------------------------------- #

    return_list = []  # 记录每个回合的return
    mean_return_list = []  # 记录每个回合的return均值

    env_test = Env(S_dim, A_dim, A, Request_number, Stop_number)
    state, _ = env_test.reset()
    while True:
        action_prob = agent.take_action(state)
        action = np.argsort(action_prob)[0][-n_actions:]
        s, _, d, _, _ = env_test.step(action)
        if d:
            break
    env_test.close()

    for i in range(max_episode):  # 迭代
        episode_return = 0  # 累计每条链上的reward
        state = env.reset()[0]  # 初始时的状态
        done = False  # 回合结束标记
        steps = 0  # 记录每个回合的步数
        while not done and steps < episode_steps:
            # 获取当前状态对应的动作概率分布 [1, n_actions]
            action_prob = agent.take_action(state)
            action = np.argsort(action_prob[0])[-A_number:]
            # 环境更新
            next_state, reward, done, _, _ = env.step(action)
            # 更新经验回放池
            replay_buffer.add(state, action_prob[0], reward, next_state, done)
            # 状态更新
            state = next_state
            # 累计每一步的reward
            episode_return += reward

            # 如果经验池超过容量，开始训练
            if replay_buffer.size() > buffer_min_size:
                # 经验池随机采样batch_size组
                s, a, r, ns, d = replay_buffer.sample(buffer_batch_size)
                # 构造数据集
                transition_dict = {
                    'states': s,
                    'action': a,
                    'rewards': r,
                    'next_states': ns,
                    'dones': d,
                }
                # 模型训练
                agent.update(transition_dict)

            # 步数加一
            steps += 1

        # 保存每一个回合的回报
        return_list.append(episode_return)
        mean_return_list.append(np.mean(return_list[-10:]))  # 平滑

        # 打印回合信息
        print(f'episode:{i}, return:{episode_return}, mean_return:{np.mean(return_list[-10:])}')

    print("Init network %f" % (env_test.cache / env_test.total))
    env_test = Env(S_dim, A_dim, A, Request_number, Stop_number)
    state, _ = env_test.reset()
    while True:
        action_prob = agent.take_action(state)
        action = np.argsort(action_prob)[0][-A_number:]
        s, _, d, _, _ = env_test.step(action)
        if d:
            break
    env_test.close()
    print("Last network %f" % (env_test.cache / env_test.total))
    agent.save_model()


def test():
    # -------------------------------------- #
    # 模型加载
    # -------------------------------------- #
    env = gym.make(env_name, render_mode="human")
    n_states = env.observation_space.shape[0]  # 状态数 2
    n_actions = env.action_space.shape[0]  # 动作数 1
    action_bound = env.action_space.high[0]  # 动作的最大值 1.0
    # 模型实例化
    agent = Agent(n_states=n_states, n_hidden=n_hidden, n_actions=n_actions, action_bound=action_bound,
                  sigma=sigma, actor_lr=actor_lr, critic_lr=critic_lr, tau=tau, gamma=gamma, device=device, path=path)
    # -------------------------------------- #
    # 模型测试
    # -------------------------------------- #
    state = env.reset()[0]  # 初始时的状态
    done = False  # 回合结束标记
    while not done:
        # 获取当前状态对应的动作 [1, n_actions]
        action = agent.take_action(state)
        # 环境更新
        next_state, _, done, _, _ = env.step(action[0])
        # 状态更新
        state = next_state

    # 关闭动画窗格
    env.close()


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# -------------------------------------- #
# 环境加载
# -------------------------------------- #

path = "../Result/checkpoints"

max_episode = 2000  # 最大回合数
episode_steps = 100  # 每回合最大步数
n_hidden = 256  # 隐含层数
sigma = 0.1  # 高斯噪声
actor_lr = 1e-1  # 策略网络学习率
critic_lr = 1e-1  # 价值网络学习率
tau = 0.5  # 软更新系数
gamma = 0.9  # 折扣因子
buffer_size = 20000  # 经验回放池容量
buffer_min_size = 64  # 经验回放池最小容量
buffer_batch_size = 32  # 经验回放池采样批次大小

A_dim = 400  # 缓存内容索引大小
S_dim = 20  # 缓存空间大小
A_number = 20  # 缓存空间大小
Request_number = 1000  # 一次请求的请求数量
A = 0.6
Stop_number = 1000  # 环境请求最大数量

env = Env(S_dim, A_dim, A, Request_number, Stop_number)

n_states = S_dim  # 状态数
n_actions = A_dim  # 动作数
action_bound = 1  # 动作的最大值 1.0

if __name__ == "__main__":
    train()
    # test()
