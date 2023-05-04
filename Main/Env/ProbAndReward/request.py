import numpy as np
from Main.Env.ProbAndReward.probability import ProbabilityDensity, ProbabilityMass


class Request:
    def __init__(self, state_space, request_number):
        self.state_space = state_space
        self.request_number = request_number
        self.request = []
        self.time_out = []
        self.time_out_stander = []
        # self.mu = 1.2
        # self.sigma = 1
        # self.class_sigma = 10
        # self.lam = 2.2
        self.a = 0.6
        # 时延范围
        self.time_out_range = [10, 10000]

        # 标准时延
        for i in range(len(self.state_space)):
            time_mu = ProbabilityMass.Poisson(i, self.lam)
            # 归一化到[0, self.time_out_range[1]]间
            time_mu = time_mu * (self.time_out_range[1] - self.time_out_range[0])
            time_mu = int(np.round(time_mu))
            # 时延误差概率分布
            self.time_out_stander.append(time_mu + self.time_out_range[0])

    def RequestCreate(self):
        distribution = ProbabilityDensity.Zipf(np.arange(len(self.state_space)),
                                               self.a, len(self.state_space))
        distribution = distribution / sum(distribution)
        # 按照概率分布生成n个请求
        requests = []
        while len(requests) < self.request_number:
            index = np.random.choice(np.arange(len(self.state_space)), p=distribution)
            requests.append(index)
        self.request = requests

    def RequestTimeOut(self):
        requests_time_out = []
        for i in range(len(self.request)):
            # time_mu = ProbabilityMass.Poisson(self.request[i], self.lam)


            # 归一化到[0, self.time_out_range[1]]间
            time_mu = time_mu * (self.time_out_range[1] - self.time_out_range[0])
            time_mu = int(np.round(time_mu))

            # # 时延误差概率分布
            # time_error = ProbabilityDensity.Normal(np.arange(self.time_out_range[1] - self.time_out_range[0]),
            #                                        time_mu, self.class_sigma)
            # time_error = time_error / sum(time_error)
            #
            # # 时延误差  0~(self.time_out_range[1] - self.time_out_range[0])
            # time_out = np.random.choice(np.arange(self.time_out_range[1] - self.time_out_range[0]), p=time_error)
            #
            # # 归一化到[self.time_out_range[0], self.time_out_range[1]]间
            # requests_time_out.append(int(time_out + self.time_out_range[0]))
            requests_time_out.append(time_mu + self.time_out_range[0])
        self.time_out = requests_time_out



if __name__ == '__main__':
    states = np.arange(5)
    request_number = 50
    request = Request(states, request_number)
    request.RequestCreate()
    request.RequestTimeOut()

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10), dpi=200)
    plt.subplot(411)
    plt.plot(range(request_number), request.request, '-', color='r')
    plt.subplot(412)
    plt.plot(range(request_number), request.time_out, '-', color='b')

    # 统计请求的分布
    index = request.request
    count = np.zeros(len(states))
    for i in range(len(states)):
        count[i] = index.count(i)
    plt.subplot(413)
    plt.bar(range(len(request.state_space)), count, color='g')

    # 统计时延的分布
    requests_time_out = []
    for i in range(len(request.state_space)):
        time_percent = ProbabilityMass.Poisson(i, request.lam)
        # 归一化到[10, 10000]间
        time_out = time_percent * (request.time_out_range[1] - request.time_out_range[0]) + request.time_out_range[0]
        requests_time_out.append(time_out)
    plt.subplot(414)
    plt.bar(range(len(request.state_space)), requests_time_out, color='y')

    plt.show()
