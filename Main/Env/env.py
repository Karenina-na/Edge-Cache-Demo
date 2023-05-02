import gym
from Main.Env.ProbAndReward.probability import ProbabilityDensity
import numpy as np
from Main.Env.ProbAndReward.Request import Request


class Env(gym.Env):
    def __init__(self, S_dim, a_dim, request_number, stop_number):
        self.observation_space = np.zeros(shape=(2, S_dim))
        self.action_space = np.arange(a_dim)
        self.request = Request(range(S_dim), request_number)
        self.request_number = request_number  # 每次生成的请求数量
        self.cache = 0
        self.total = 0
        self.stop_number = stop_number  # 仿真step步数
        self.request_time_out_dis = [[] for _ in range(S_dim)]  # 时延分布

    def step(self, action):
        # 计算缓存命中率，即奖励
        reward = 0
        for req in self.request.request:
            if req in action:
                reward += 1
                self.cache += 1
            else:
                reward += -1
            self.total += 1
        # 生成新的请求
        self.request.RequestCreate()
        self.request.RequestTimeOut()

        # observation_space更新 [频率，时延均值] [2, S_dim]
        frequency = np.zeros(shape=(len(self.observation_space[0])))
        time_out = np.zeros(shape=(len(self.observation_space[1])))
        for i in range(len(self.observation_space[0])):
            frequency[i] += self.request.request.count(i)
        for index in self.request.request:
            time_out[index] += self.request.time_out[index]
        self.observation_space[0] = frequency
        self.observation_space[1] = time_out / self.request_number

        # request_time_out_dis更新 [状态index, 时延]  [S_dim, S_dim]
        print(self.request_time_out_dis)
        exit(0)
        for index in self.request.request:
            self.request_time_out_dis[index].append(self.request.time_out[index])


        # 结束条件
        if self.total >= self.stop_number:
            return self.observation_space, reward, True, False, False
        return self.observation_space, reward, False, False, False

    def reset(self):
        # 生成新的请求
        self.request.RequestCreate()
        self.request.RequestTimeOut()

        # observation_space更新
        frequency = np.zeros(shape=(len(self.observation_space[0])))
        for i in range(len(self.observation_space[0])):
            frequency[i] += self.request.request.count(i)
        time_out = np.zeros(shape=(len(self.observation_space[1])))
        for index in self.request.request:
            time_out[index] += self.request.time_out[index]
        self.observation_space[0] = frequency
        self.observation_space[1] = time_out / self.request_number

        # request_time_out_dis更新 [状态index, 时延]  [S_dim, S_dim]
        for index in self.request.request:
            self.request_time_out_dis[index].append(self.request.time_out[index])

        self.cache = 0
        self.total = 0
        return self.observation_space, False


if __name__ == "__main__":
    s_dim = 10
    a_dim = 5
    request_number = 50
    stop_number = 100
    env = Env(s_dim, a_dim, request_number, stop_number)
    count = np.zeros(shape=s_dim)
    time_out = np.zeros(shape=s_dim)
    time_out_dis = np.zeros(shape=(s_dim, s_dim))
    env.reset()
    while True:
        # observation_space [频率，时延均值] [2, S_dim]
        observation_space, _, done, _, _ = env.step(np.arange(a_dim))
        request_error_dis = env.request_time_out_dis
        print(request_error_dis)
        exit(0)
        for i in range(s_dim):
            count[i] += observation_space[0][i]
            time_out[i] += observation_space[1][i]
            time_out[i] /= 2  # 时延均值

        if done:
            break
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, s_dim * 3), dpi=200)
    plt.subplots_adjust(hspace=0.5)
    for i in range(s_dim):
        plt.subplot(s_dim, 1, i + 1)
        plt.plot(range(s_dim), count[i], '-', color='r')
        plt.ylabel('count')
        plt.xlabel('step')
        plt.title('state %d' % i)
    plt.show()
