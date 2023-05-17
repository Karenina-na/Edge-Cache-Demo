import gym
from itertools import combinations
from param import *
from Env.module.probability import *


# 动作空间
class ActionSpace:
    def __init__(self, cache_cab, file_number):
        """
        :param cache_cab: 缓存能力
        :param file_number: 文件数量
        """
        self.action_space = np.arange(file_number)  # 动作空间索引长度

        self.action_index_dic = []  # 存储编号-枚举的动作
        ## 缓存循环组合数
        # comb = combinations(self.action_space, cache_cab)
        # for i in comb:
        #     self.action_index_dic.append(i)
        # 缓存索引循环位移
        cache = [i for i in range(S_dim)]
        for i in range(file_number):
            self.action_index_dic.append(cache[:cache_cab])
            cache = cache[1:] + cache[:1]
        np.random.shuffle(self.action_index_dic)

        self.actions_index_number = len(self.action_index_dic)  # 动作空间的大小


class Env(gym.Env):
    def __init__(self):
        # [0]内容流行度，[1]上一次缓存的内容
        self.observation_space = np.zeros(shape=(2, S_dim))
        # 生成的请求
        self.request = np.zeros(shape=Request_number)
        # 动作空间
        self.action_space = ActionSpace(cache_space, S_dim)
        # 每次生成的请求数量
        self.request_number = Request_number
        # 仿真step步数
        self.stop_number = Stop_number
        # 内容流行度分布
        self.distribution = ProbabilityDensity.Zipf(np.arange(S_dim), Zipf_alpha, S_dim)
        # 内容流行度记请求次数改变
        self.request_time = 0

    def step(self, action):
        """
        action: 缓存动作索引编号
        example: 34
        """
        # 将动作转成具体的文件索引
        cache_index = self.action_space.action_index_dic[action]
        # 计算缓存命中率
        cache_hit = 0  # 缓存命中率
        for request in self.request:
            if request in cache_index:
                cache_hit += 1
        # 请求中非 -1 的数量
        cache_total = len(self.request[self.request != -1])

        # 计算奖励，缓存命中率越高，奖励越高
        # reward = cache_hit / cache_total
        reward = cache_hit

        # 统计上一时刻请求频率
        last_time_request = np.zeros(S_dim)
        for index in self.request:
            last_time_request[int(index)] += 1

        # 生成新的请求
        self.request, content_popularity = self.CreateRequest()

        # 将现在缓存的内容转为one-hot编码
        now_cache = np.zeros(S_dim)
        for cache in cache_index:
            now_cache[int(cache)] = 1

        # 更新状态
        self.observation_space[0] = content_popularity * 10
        self.observation_space[1] = now_cache
        # self.observation_space[2] = last_time_request

        # 判断是否结束
        self.stop_number -= Request_number
        self.request_time += Request_number
        d = self.stop_number <= 0

        return self.observation_space.reshape(-1), reward, d, \
            {"cache_hit": cache_hit, "cache_total": cache_total}, False

    def reset(self):
        # 初始化状态
        request, content_popularity = self.CreateRequest()
        self.observation_space[0] = content_popularity
        self.observation_space[1] = np.zeros(S_dim)
        # self.observation_space[2] = np.zeros(S_dim)
        self.request = request

        # 初始化仿真步数
        self.stop_number = Stop_number
        self.distribution = ProbabilityDensity.Zipf(np.arange(S_dim), Zipf_alpha, S_dim)
        self.request_time = 0

        return self.observation_space.reshape(-1), {}

    def CreateRequest(self):
        """
        生成请求
        :return: 请求，内容流行度
        request = [1,2,3,4,5,6,7,8,9,10]
        """

        # 打乱顺序，生成节点的请求
        request = np.zeros(Request_number)  # 请求
        content_popularity = np.zeros(S_dim)  # 内容流行度
        # 内容流行度移位
        if self.request_time >= w:
            self.request_time = 0
            # 循环右移一位
            self.distribution = np.concatenate(
                (self.distribution[-1:], self.distribution[:-1]))
            self.distribution = self.distribution / sum(self.distribution)
        content_popularity = self.distribution
        # 按概率分布生成固定次数的请求
        req = []
        for i in range(S_dim):
            for j in range(int(Request_number * content_popularity[i])):
                req.append(i)
        # 维度对齐
        while len(req) < Request_number:
            req.append(-1)
        self.request = np.array(req[:Request_number])
        return self.request, content_popularity


if __name__ == "__main__":
    # 将文件运行根目录改为"../"
    import os

    os.chdir("../")

    env = Env()
    obs, _ = env.reset()

    action = 2

    obs, reward, done, info, _ = env.step(action)
    print("action size", np.array(action).shape)
    print("state size ", np.array(obs).shape)
    print("reward ", reward)
    print("done ", done)
