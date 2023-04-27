from Env.env import  Env
import numpy as np
import torch
import torch.nn as nn
import random
from Agent.DQN_Agent import Agent, ActionSpace


def train():
    env = Env(S_dim, A_dim, A, Request_number, Stop_number)
    action_space = ActionSpace(A_number, A_dim)

    s, info = env.reset()

    n_state = S_dim
    n_action = action_space.n_action
    """Generate agents"""

    agent = Agent(idx=0, n_input=n_state, n_output=n_action, mode='train', model_path=model_path)

    """Main Training Loop"""

    REWARD_BUFFER = np.zeros(shape=n_episode)

    # 初始
    agent_test = Agent(idx=0, n_input=n_state, n_output=n_action, mode='train')
    env_test = Env(S_dim, A_dim, A, Request_number, Stop_number)
    s, _ = env_test.reset()
    while True:
        a_index = agent_test.online_net.act(s)
        a = action_space.dic[a_index]
        s, _, d, _, _ = env_test.step(a)
        if d:
            break
    env_test.close()

    for episode_i in range(n_episode):

        # 一次游戏的总reward
        episode_reward = 0

        for step_i in range(n_time_step):

            # 选择随机动作的概率
            epsilon = np.interp(episode_i * n_time_step + step_i, [0, EPSILON_DECAY],
                                [EPSILON_START, EPSILON_END])  # interpolation

            # 概率epsilon随机选择动作，概率1-epsilon选择最优动作 online网络
            random_sample = random.random()
            if random_sample <= epsilon:
                max_index = len(action_space.dic) - 1
                a_index = random.randint(0, max_index)
                a = action_space.dic[a_index]
            else:
                a_index = agent.online_net.act(s)
                a = action_space.dic[a_index]

            # 执行动作
            s_, r, done, trunk, info = env.step(a)  # trunk,info will not be used

            # 记录经验
            agent.memo.add_memo(s, a_index, r, done, s_)

            # 更新状态
            s = s_

            # 记录奖励
            episode_reward += r

            if done:
                s, info = env.reset()

                # 记录奖励
                REWARD_BUFFER[episode_i] = episode_reward
                break

            # 抽取经验
            batch_s, batch_a, batch_r, batch_done, batch_s_ = agent.memo.sample()  # update batch-size amounts of Q

            # 计算 target Q(t+1)
            target_q_values = agent.target_net(batch_s_)
            # 计算 target max Q(t+1)
            max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

            # 计算 target Q(t)
            targets = batch_r + agent.GAMMA * (1 - batch_done) * max_target_q_values

            # 计算 online Q(t)
            q_values = agent.online_net(batch_s)

            # 按照 batch_a 选择 Q(t) 中的动作 维度为行
            a_q_values = torch.gather(input=q_values, dim=1, index=batch_a)

            # 计算 loss
            loss = nn.functional.smooth_l1_loss(a_q_values, targets)

            # 更新 online 网络
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()

        # 更新 target 网络
        if episode_i % TARGET_UPDATE_FREQUENCY == 0 and episode_i != 0:
            agent.target_net.load_state_dict(agent.online_net.state_dict())

            print("Episode: {}".format(episode_i))
            print("Avg Reward: {}".format(np.mean(REWARD_BUFFER[:episode_i])))

    agent.save_model()

    print("game over")
    print("Init network %f" % (env_test.cache / env_test.total))

    env_test = Env(S_dim, A_dim, A, Request_number, Stop_number)
    s, _ = env_test.reset()
    while True:
        a_index = agent.online_net.act(s)
        a = action_space.dic[a_index]
        s, _, d, _, _ = env_test.step(a)
        if d:
            break
    env_test.close()
    print("last network %f" % (env_test.cache / env_test.total))




def test():
    """Generate agents"""
    env = Env(S_dim, A_dim, A, Request_number, Stop_number)
    s, info = env.reset()
    action_space = ActionSpace(A_number, A_dim)

    s, info = env.reset()

    n_state = S_dim
    n_action = action_space.n_action

    agent = Agent(idx=0, n_input=n_state, n_output=n_action, mode='test', model_path=model_path)

    env_test = Env(S_dim, A_dim, A, Request_number, Stop_number)
    s, _ = env_test.reset()
    while True:
        a_index = agent.online_net.act(s)
        a = action_space.dic[a_index]
        s, _, d, _, _ = env_test.step(a)
        if d:
            break
    env_test.close()

    print("cache hit ratio %f" % (env_test.cache / env_test.total))


# 探索率初始
EPSILON_START = 0.5
# 探索率结束
EPSILON_END = 0.02
# 探索率衰减率
EPSILON_DECAY = 100000
# Target Network 更新频率
TARGET_UPDATE_FREQUENCY = 10
# 训练次数
n_episode = 200
# 每次训练的最大步数
n_time_step = 1000
# model
model_path = "../Result/checkpoints"


A_dim = 30  # 缓存内容索引大小
S_dim = 4  # 缓存空间大小
A_number = 4  # 缓存空间大小
Request_number = 10  # 一次请求的请求数量
A = 0.6
Stop_number = 10000  # 环境请求最大数量

if __name__ == '__main__':
    # train()
    test()
