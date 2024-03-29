import numpy as np
import torch
import torch.nn as nn
import random
from Agent.DQN_Agent import Agent
from Env.env import Env
from Param import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    env = Env()
    s, info = env.reset()
    n_state = s.shape[0]
    n_action = env.action_space.actions_index_number
    """Generate agents"""

    agent = Agent(idx=0, n_input=n_state, n_output=n_action, mode='train', model_path=model_path, device=device)

    """Main Training Loop"""

    REWARD_BUFFER = np.zeros(shape=n_episode)
    HIT_RATE = []
    TIME_OUT = []
    BASELINE = []
    REWARD = []
    for episode_i in range(n_episode):

        # 一次游戏的总reward
        episode_reward = 0
        for step_i in range(n_time_step):

            # 选择随机动作的概率
            epsilon = max(EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END) * (step_i / EPSILON_DECAY))

            # 概率epsilon随机选择动作，概率1-epsilon选择最优动作 online网络
            random_sample = random.random()
            if random_sample <= epsilon:
                a = np.random.randint(0, n_action)
            else:
                a = agent.online_net.act(s)
            # 执行动作
            s_, r, done, info, _ = env.step(a)  # trunk,info will not be used

            # 记录经验
            agent.memo.add_memo(s, a, r, done, s_)

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
            batch_s = torch.as_tensor(batch_s, dtype=torch.float32).to(device)
            batch_a = torch.as_tensor(batch_a, dtype=torch.int64).to(device)
            batch_r = torch.as_tensor(batch_r, dtype=torch.float32).to(device)
            batch_done = torch.as_tensor(batch_done, dtype=torch.float32).to(device)
            batch_s_ = torch.as_tensor(batch_s_, dtype=torch.float32).to(device)

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
            # 演示
            env_test = Env()
            s, info = env_test.reset()
            reward = 0
            total = 0
            hit = 0
            baseline = 0
            time_out = 0
            step = 0
            agent.online_net.eval()
            while True:
                s = torch.as_tensor(s, dtype=torch.float32).unsqueeze(0)
                a = agent.online_net.act(s)
                s, r, done, info, _ = env_test.step(a)
                total += info["cache_total"]
                hit += info["cache_hit"]
                baseline += info["baseline_count"]
                time_out += info["time_out"]
                reward += r
                step += 1
                if done or step >= n_time_step:
                    break
            print("Hit Rate: ", hit / total)
            print("Avg Baseline: ", baseline / step)
            print("Avg Time Out: ", time_out / step)
            print("Avg Reward: ", reward/step)
            HIT_RATE.append(hit / total)
            TIME_OUT.append(time_out / step)
            BASELINE.append(baseline / step)
            REWARD.append(reward/step)
            print("#" * 50)
            # record

            agent.online_net.train()

    env = Env()
    s, info = env.reset()
    reward = 0
    total = 0
    hit = 0
    time_out = 0
    # 演示
    step = 0
    while True:
        s = torch.as_tensor(s, dtype=torch.float32).unsqueeze(0)
        a = agent.online_net.act(s)
        s, r, done, info, _ = env.step(a)
        total += info["cache_total"]
        hit += info["cache_hit"]
        time_out += info["time_out"]
        reward += r
        step += 1
        if done or step >= n_time_step:
            env.reset()
            print("#" * 50)
            print("Finished Reward: ", reward)
            print("Step Number: ", step)
            print("Hit Rate: ", hit / total)
            print("Time Out: ", time_out / step)
            print("-" * 50)
            break

    # 保存数据
    import pandas as pd
    reward_all = np.array(REWARD)
    time_out_all = np.array(TIME_OUT)
    hit_rate_all = np.array(HIT_RATE)
    baseline_all = np.array(BASELINE)
    print(reward_all.shape)
    print(time_out_all.shape)
    print(hit_rate_all.shape)
    print(baseline_all.shape)
    reward_pd = pd.DataFrame(reward_all)
    time_out_pd = pd.DataFrame(time_out_all)
    hit_rate_pd = pd.DataFrame(hit_rate_all)
    baseline_pd = pd.DataFrame(baseline_all)
    reward_pd.to_csv("./Result/logs/reward.csv", index=False, header=False)
    time_out_pd.to_csv("./Result/logs/time_out.csv", index=False, header=False)
    hit_rate_pd.to_csv("./Result/logs/hit_rate.csv", index=False, header=False)
    baseline_pd.to_csv("./Result/logs/baseline.csv", index=False, header=False)


if __name__ == '__main__':
    train()
    # test()
