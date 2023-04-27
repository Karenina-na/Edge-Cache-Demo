import gym
from Main.Env.ProbAndReward.probability import ProbabilityDensity
import numpy as np


class Env(gym.Env):
    def __init__(self, S_dim, a_dim, a, request_number, stop_number):
        self.observation_space = np.ones(S_dim)
        self.action_space = np.arange(a_dim)
        self.a = a
        self.request = []
        self.request_number = request_number
        self.cache = 0
        self.total = 0
        self.stop_number = stop_number

    def step(self, action):

        # 计算缓存命中率，即奖励
        reward = 0
        for req in self.request:
            if req in action:
                reward += 1
                self.cache += 1
            else:
                reward += -1
            self.total += 1
        # 消除第一个维度
        self.observation_space = np.squeeze(action)

        # 生成新的请求
        # -----------------------------------------------------------------------------distribution#
        distribution = ProbabilityDensity.Zipf(np.arange(len(self.observation_space)),
                                               self.a, len(self.observation_space))
        # 归一化
        distribution = distribution / sum(distribution)
        # 按照概率分布生成n个请求
        requests = []
        while len(requests) <= self.request_number:
            index = np.random.choice(np.arange(len(self.observation_space)), p=distribution)
            requests.append(index)
        self.request = requests

        # 结束条件
        if self.total >= self.stop_number:
            return self.observation_space, reward, True, False, False
        return self.observation_space, reward, False, False, False

    def reset(self):
        # -----------------------------------------------------------------------------distribution#
        distribution = ProbabilityDensity.Zipf(np.arange(len(self.observation_space)),
                                               self.a, len(self.observation_space))
        # 归一化
        distribution = distribution / sum(distribution)
        # 按照概率分布生成n个请求
        requests = []
        while len(requests) <= self.request_number:
            index = np.random.choice(np.arange(len(self.observation_space)), p=distribution)
            requests.append(index)
        self.request = requests
        self.cache = 0
        self.total = 0
        return self.observation_space, False

    def render(self):
        pass

    def close(self):
        pass