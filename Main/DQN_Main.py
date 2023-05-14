import numpy as np
import torch
import torch.nn as nn
import random
from Agent.DQN_Agent import Agent
from Main.Env.env import Env
from Main.Env.param import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    env = Env()

    s, info = env.reset()
    s = np.swapaxes(s, 0, 1)
    s = np.reshape(s, newshape=(len(s), -1))
    n_state = s.shape[1]
    n_action = env.action_space.actions_index_number
    agent = Agent(idx=0, n_input=n_state, n_output=n_action, mode='train', model_path=model_path, device=device)

    cache_hit_init = [0, 0, 0]
    cache_total_init = [0, 0, 0]
    node_time_out_init = [0, 0, 0]
    # 玩游戏
    env_init = Env()
    s, info = env.reset()
    while True:
        s = np.swapaxes(s, 0, 1)
        s = np.reshape(s, newshape=(len(s), -1))
        a = []
        s = torch.tensor(s, dtype=torch.float32, device=device)
        for index in s:
            a.append(agent.online_net.act(index))
        s, r, done, info, _ = env_init.step(a)  # trunk,info will not be used
        cache_hit_init += np.reshape(info["cache_hit"], newshape=(Node_number,))
        cache_total_init += np.reshape(info["cache_total"], newshape=(Node_number,))
        node_time_out_init += np.reshape(info["node_timeout"], newshape=(Node_number,))
        if done:
            break

    """Main Training Loop"""

    REWARD_BUFFER = np.zeros(shape=n_episode)

    for episode_i in range(n_episode):
        s, info = env.reset()
        s = np.swapaxes(s, 0, 1)
        s = np.reshape(s, newshape=(len(s), -1))

        # 一次游戏的总reward
        episode_reward = 0

        for step_i in range(n_time_step):
            s = torch.tensor(s, dtype=torch.float32, device=device)
            # 选择随机动作的概率
            epsilon = max(EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END) * (step_i / EPSILON_DECAY))

            # 概率epsilon随机选择动作，概率1-epsilon选择最优动作 online网络
            random_sample = random.random()
            if random_sample <= epsilon:
                a = []
                for i in range(len(s)):
                    a.append(np.random.randint(n_action))
            else:
                a = []
                for index in s:
                    a.append(agent.online_net.act(index))
            s = s.cpu()

            # 执行动作
            s_, r, done, trunk, info = env.step(a)  # trunk,info will not be used
            # 记录经验
            s_ = np.swapaxes(s_, 0, 1)
            s_ = np.reshape(s_, newshape=(len(s), -1))
            for i in range(len(s)):
                agent.memo.add_memo(s[i], a[i], r[i], done, s_[i])

            # 更新状态
            s = s_

            # 记录奖励
            for i in range(len(r)):
                episode_reward += r[i]
            if done:
                s, info = env.reset()
                # 记录奖励
                REWARD_BUFFER[episode_i] = episode_reward
                break

            # 抽取经验
            batch_s, batch_a, batch_r, batch_done, batch_s_ = agent.memo.sample()  # update batch-size amounts of Q
            batch_s = batch_s.to(device)
            batch_a = batch_a.to(device)
            batch_r = batch_r.to(device)
            batch_done = batch_done.to(device)
            batch_s_ = batch_s_.to(device)

            # 计算 target Q(t+1) target网络
            target_q_values = agent.target_net(batch_s_)

            # 计算 target max Q(t+1)
            max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

            # 计算 target Q(t)
            targets = batch_r + agent.GAMMA * (1 - batch_done) * max_target_q_values

            # 计算 online Q(t) online网络
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

    # 玩游戏
    env_last = Env()
    s, info = env.reset()
    cache_hit = [0, 0, 0]
    cache_total = [0, 0, 0]
    node_time_out = [0, 0, 0]
    while True:
        s = np.swapaxes(s, 0, 1)
        s = np.reshape(s, newshape=(len(s), -1))
        a = []
        s = torch.tensor(s, dtype=torch.float32, device=device)
        for index in s:
            a.append(agent.online_net.act(index))
        s, r, done, info, _ = env_last.step(a)  # trunk,info will not be used
        cache_hit += np.reshape(info["cache_hit"], newshape=(Node_number,))
        cache_total += np.reshape(info["cache_total"], newshape=(Node_number,))
        node_time_out += np.reshape(info["node_timeout"], newshape=(Node_number,))
        if done:
            break

    for i in range(Node_number):
        print("node %d cache hit ratio %f init" % (i, cache_hit_init[i] / cache_total_init[i]))
    print()
    for i in range(Node_number):
        print("node %d cache time out %f init" % (i, node_time_out_init[i] / cache_total_init[i]))
    print('-' * 100)
    for i in range(Node_number):
        print("node %d cache hit ratio %f" % (i, cache_hit[i] / cache_total[i]))
    print()
    for i in range(Node_number):
        print("node %d cache time out %f" % (i, node_time_out[i] / cache_total[i]))
    # agent.save_model()


def test():
    """Generate agents"""
    env = Env()

    s, info = env.reset()
    s = np.swapaxes(s, 0, 1)
    s = np.reshape(s, newshape=(len(s), -1))
    n_state = s.shape[1]
    n_action = env.action_space.actions_index_number
    agent = Agent(idx=0, n_input=n_state, n_output=n_action, mode='train', model_path=model_path)
    reward_all = 0

    # 演示
    s, info = env.reset()
    step = 0
    while True:
        s = np.swapaxes(s, 0, 1)
        s = np.reshape(s, newshape=(len(s), -1))
        a = []
        for index in s:
            a.append(agent.online_net.act(index))
        s, r, done, trunk, info = env.step(a)
        reward_all = 0
        for i in range(len(r)):
            reward_all += r[i]
        step += 1
        if done or step >= n_time_step:
            env.reset()
            print("#" * 50)
            print("Finished Reward: ", reward_all)
            print("Step Number: ", step)
            print("-" * 50)
            break


# 探索率初始
EPSILON_START = 1.0
# 探索率结束
EPSILON_END = 0.02
# 探索率衰减率
EPSILON_DECAY = 0.0001
# Target Network 更新频率
TARGET_UPDATE_FREQUENCY = 10
# 平均reward到达多少演示
DEMO_REWARD = 100
# 训练次数
n_episode = 500
# 每次训练的最大步数
n_time_step = 500
# 游戏
# model
model_path = None

if __name__ == '__main__':
    train()
    # test()
