import gym
from Main.Env.ProbAndReward.probability import ProbabilityDensity, ProbabilityMass
import numpy as np
from Main.Env.param import *
from itertools import combinations


# 动作空间
class ActionSpace:
    def __init__(self, cache_cab, file_number):
        """
        :param cache_cab: 缓存能力
        :param file_number: 文件数量
        """
        self.action_space = np.arange(file_number)  # 动作空间索引长度

        self.action_index_dic = []  # 存储编号-枚举的动作

        comb = combinations(self.action_space, cache_cab)
        for i in comb:
            self.action_index_dic.append(i)
        np.random.shuffle(self.action_index_dic)

        self.actions_index_number = len(self.action_index_dic)  # 动作空间的大小


class Env(gym.Env):
    def __init__(self):
        # [0]不同节点的内容流行度，[1]不同节点上一次缓存的内容
        self.observation_space = np.zeros(shape=(2, Node_number, A_dim))
        # 生成的请求
        self.request = np.zeros(shape=(Node_number, Request_number))
        # 动作空间
        self.action_space = ActionSpace(A_number, A_dim)
        # 每次生成的请求数量
        self.request_number = Request_number
        # 仿真step步数
        self.stop_number = Stop_number
        # 计算每个节点的平均传输时间延迟
        self.node_timeout = np.zeros(shape=(Node_number, 1))
        # 传输时间分布参数
        self.Lambda = 1

    def step(self, action):
        """
        action: [节点编号，缓存动作索引编号]
        example: [34,12,35] if Node_number = 3
        """
        # 将动作转成具体的文件索引
        cache_index = []
        for node in action:
            cache_index.append(self.action_space.action_index_dic[node])
        print(cache_index)
        # 计算缓存命中率
        cache_hit = np.zeros(shape=(Node_number, 1))  # 缓存命中率
        cache_total = np.zeros(shape=(Node_number, 1))  # 缓存请求数量
        for i in range(Node_number):
            for request in self.request[i]:
                if request in cache_index[i]:
                    cache_hit[i] += 1
                cache_total[i] += 1

        # 计算奖励，缓存命中率越高，奖励越高
        reward_node = np.zeros(shape=(Node_number, 1))  # 节点奖励
        for i in range(Node_number):
            reward_node[i] = -(cache_total[i] - cache_hit[i])

        # 计算传输时间延迟
        # 将缓存索引转成one-hot编码
        last_cache = self.observation_space[1]
        now_cache = np.zeros(shape=(Node_number, A_dim))
        for i in range(Node_number):
            for j in range(A_dim):
                if j in cache_index[i]:
                    now_cache[i][j] = 1
        # 计算每个节点的平均传输时间延迟(卫星拉取时延)
        for i in range(Node_number):
            for file_index in range(A_dim):
                if last_cache[i][file_index] == 0 and now_cache[i][file_index] == 1:
                    self.node_timeout[i] += self.FileTransferTime(file_index, self.Lambda)
            self.node_timeout[i] /= A_dim

        # 生成新的请求
        self.request, content_popularity = self.CreateRequest()

        # 更新状态
        self.observation_space[0] = content_popularity
        self.observation_space[1] = now_cache

        # 判断是否结束
        self.stop_number -= Request_number
        done = self.stop_number <= 0

        return self.observation_space, reward_node, done, \
            {"cache_hit": cache_hit, "cache_total": cache_total,
             "node_timeout": self.node_timeout}, False

    def reset(self):
        # 初始化状态
        request, content_popularity = self.CreateRequest()
        self.observation_space[0] = content_popularity
        self.observation_space[1] = np.zeros(shape=(Node_number, A_dim))
        self.request = request

        # 初始化仿真步数
        self.stop_number = Stop_number

        # 初始化传输时间延迟
        self.node_timeout = np.zeros(shape=(Node_number, 1))
        return self.observation_space, {}

    @staticmethod
    def CreateRequest():
        """
        生成请求
        :return: 请求，内容流行度
        example: if node number is 3
        request = [
        [1,2,3,4,5,6,7,8,9,10],
        [1,2,3,4,5,6,7,8,9,10],
        [1,2,3,4,5,6,7,8,9,10],
        ]
        """
        # 不同节点的内容流行度分布生成
        distribution = ProbabilityDensity.Zipf(np.arange(A_dim), Zipf_alpha, A_dim)
        distribution = distribution / sum(distribution)
        # 打乱顺序，生成节点的请求
        request = np.zeros(shape=(Node_number, Request_number))  # 请求
        content_popularity = np.zeros(shape=(Node_number, A_dim))  # 内容流行度
        for i in range(Node_number):
            # 更改distribution的顺序，模拟节点的内容流行度分布不同
            # func(distribution)
            content_popularity[i] = distribution
            request[i] = np.random.choice(np.arange(A_dim), Request_number, p=distribution)
        return request, content_popularity

    @staticmethod
    def FileTransferTime(file_index, Lambda):
        """
        计算每个节点的平均传输时间延迟
        :param file_index: 文件索引
        :param Lambda: 传输时间分布参数
        :return: 传输时间延迟
        """
        dic = {
            i: ProbabilityMass.Poisson(i, Lambda) for i in range(A_dim)
        }
        return dic[file_index]


if __name__ == "__main__":
    env = Env()
    obs, _ = env.reset()

    action = [i + 2 for i in range(Node_number)]
    for a in action:
        print("Node ", a, " cache ", env.action_space.action_index_dic[a])

    obs, reward, done, info, _ = env.step(action)
    print("action size", np.array(action).shape)
    print("state size ", np.array(obs).shape)
    print("reward ", reward)
    print("done ", done)
