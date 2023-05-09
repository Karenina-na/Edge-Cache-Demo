import gym
from Main.Env.ProbAndReward.probability import ProbabilityDensity
import numpy as np
from Main.Env.ProbAndReward.request import Request
from Main.Env.param import *


class Env(gym.Env):
    def __init__(self, S_dim, a_dim, request_number, stop_number):
        # self.observation_space = np.zeros(shape=(3, S_dim))
        self.observation_space = np.zeros(shape=(3, S_dim))
        self.action_space = np.arange(a_dim)
        self.request = Request(range(S_dim), request_number)
        self.request_number = request_number  # 每次生成的请求数量
        self.cache = 0
        self.total = 0
        self.time_out_file = 0
        self.stop_number = stop_number  # 仿真step步数
        self.request_time_out_dis = [[] for _ in range(S_dim)]  # 时延分布

    def step(self, action):
        reward_hit = 0
        reward_time_out = 0
        for index in range(len(self.request.request)):
            self.total += 1
            print(action)
            if action[self.request.request[index]] == 1:
                # 缓存命中
                reward_hit += 1
                self.cache += 1
                if self.request.time_out[index] > self.request.time_out_max:
                    # 缓存了超时的文件
                    reward_time_out += 1
            else:
                # 缓存没命中
                reward_hit += -1
                if self.request.time_out[index] > self.request.time_out_max:
                    # 没命中且超时
                    self.time_out_file += 1
                    reward_time_out += -1

        reward = w * reward_hit + (1 - w) * reward_time_out

        # observation_space更新 [频率，时延均值, 上一时刻缓存内容] [3, S_dim]
        frequency = np.zeros(shape=(len(self.observation_space[0])))
        for i in range(len(self.observation_space[0])):
            frequency[i] += self.request.request.count(i)
        self.observation_space[0] = frequency  # 频率

        time_out = np.zeros(shape=(len(self.observation_space[1])))
        for i in range(len(self.request.request)):
            time_out[self.request.request[i]] += self.request.time_out[i]
        for index in range(len(self.observation_space[1])):
            if frequency[index] != 0:
                time_out[index] = time_out[index] / frequency[index]
        self.observation_space[1] = time_out  # 时延均值

        cache_index = np.zeros(shape=(len(self.observation_space[2])))
        for index in action:
            cache_index[index] = 1
        self.observation_space[2] = cache_index  # 上一时刻缓存内容

        # request_time_out_dis更新 [状态index, 时延]  [S_dim, S_dim]
        for i in range(len(self.request.request)):
            self.request_time_out_dis[self.request.request[i]].append(self.request.time_out[i])  # 时延分布

        # 生成新的请求
        self.request.RequestCreate()
        self.request.RequestTimeOut()

        # 结束条件
        if self.total >= self.stop_number:
            return self.observation_space, reward, True, False, False
        return self.observation_space, reward, False, False, False

    def reset(self):
        # observation_space更新 [频率，时延均值, 上一时刻缓存内容] [3, S_dim]
        frequency = np.zeros(shape=(len(self.observation_space[0])))
        for i in range(len(self.observation_space[0])):
            frequency[i] += self.request.request.count(i)
        self.observation_space[0] = frequency  # 频率

        time_out = np.zeros(shape=(len(self.observation_space[1])))
        for i in range(len(self.request.request)):
            time_out[self.request.request[i]] += self.request.time_out[i]
        for index in range(len(self.observation_space[1])):
            if frequency[index] != 0:
                time_out[index] = time_out[index] / frequency[index]
        self.observation_space[1] = time_out  # 时延均值

        cache_index = np.zeros(shape=(len(self.observation_space[2])))
        self.observation_space[2] = cache_index  # 上一时刻缓存内容

        # request_time_out_dis更新 [状态index, 时延]  [S_dim, S_dim]
        for i in range(len(self.request.request)):
            self.request_time_out_dis[self.request.request[i]].append(self.request.time_out[i])  # 时延分布

        self.cache = 0
        self.total = 0

        # 生成新的请求
        self.request.RequestCreate()
        self.request.RequestTimeOut()

        return self.observation_space, False


class ActionSpace:
    def __init__(self, n_action, action_space, ):
        self.action_space = np.arange(action_space)  # 动作空间

        self.dic = []  # 存储编号-枚举的动作

        comb = combinations(self.action_space, n_action)
        for i in comb:
            self.dic.append(i)

        self.n_action = len(self.dic)  # 动作空间的大小


if __name__ == "__main__":
    s_dim = 10
    a_dim = 10
    request_number = 200
    stop_number = 500
    env = Env(s_dim, a_dim, request_number, stop_number)
    frequency = np.zeros(shape=s_dim)
    time_out_mean = np.zeros(shape=s_dim)
    dic = {i: 0. for i in range(s_dim)}
    time_out_dis = [[] for _ in range(s_dim)]
    step = 0
    # 初始化
    observation_space, _ = env.reset()
    for i in range(s_dim):
        frequency[i] += observation_space[0][i]
        time_out_mean[i] += observation_space[1][i]
        if observation_space[0][i] != 0:
            dic[i] += 1
    while True:
        # observation_space [频率，时延均值, 上一时刻缓存状态] [3, S_dim]
        L = np.zeros(shape=len(env.request_time_out_dis), dtype=int).tolist()
        observation_space, _, done, _, _ = env.step(L)
        request_error_dis = env.request_time_out_dis
        for i in range(s_dim):
            frequency[i] += observation_space[0][i]
            time_out_mean[i] += observation_space[1][i]
            if observation_space[0][i] != 0:
                dic[i] += 1
            time_out_dis[i] += request_error_dis[i]
        if done:
            break
    for i in range(s_dim):
        if dic[i] != 0:
            time_out_mean[i] = time_out_mean[i] / dic[i]
    # 画图
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, s_dim * 3), dpi=200)
    plt.subplots_adjust(hspace=0.5)
    for i in range(s_dim):
        plt.subplot(s_dim + 1, 1, i + 1)
        plt.plot(range(len(time_out_dis[i])), time_out_dis[i], '-', color='r')
        plt.ylabel('time out')
        plt.xlabel('step')
        plt.title("number %d file's time out distribution" % i)
        # 理论均值线
        plt.axhline(y=env.request.time_out_stander[i], color="b", linestyle="-")
        plt.text(0, env.request.time_out_stander[i], 'stander')
        # 实际均值线
        plt.axhline(y=time_out_mean[i], color="g", linestyle="-")
        plt.text(0, time_out_mean[i], 'mean')
    plt.subplot(s_dim + 1, 1, s_dim + 1)
    plt.scatter(range(len(frequency)), frequency, marker='o', color='b')
    plt.xticks(range(len(frequency)))
    plt.ylabel('frequency')
    plt.xlabel('file')
    plt.title("frequency")
    plt.show()
