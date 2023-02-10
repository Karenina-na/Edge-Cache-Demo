import torch
import numpy as np
import os
import gym

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Define the probability density function
class ProbabilityDensity(object):
    @staticmethod
    def StandardNormal(x):
        """Standard normal distribution"""
        return 1 / np.sqrt(2 * np.pi) * np.exp(-x ** 2 / 2)

    @staticmethod
    def Normal(x, mu, sigma):
        """Normal distribution"""
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    @staticmethod
    def Exponential(x, lam):
        """Exponential distribution"""
        return lam * np.exp(-lam * x)

    @staticmethod
    def Beta(x, alpha, beta):
        """Beta distribution"""
        return x ** (alpha - 1) * (1 - x) ** (beta - 1) / (
                np.math.gamma(alpha) * np.math.gamma(beta) / np.math.gamma(alpha + beta))


# Define the probability mass function
class ProbabilityMass(object):

    @staticmethod
    def Bernoulli(x, p):
        """Bernoulli distribution"""
        return p ** x * (1 - p) ** (1 - x)

    @staticmethod
    def Binomial(x, n, p):
        """Binomial distribution"""
        return np.math.factorial(n) / (np.math.factorial(x) * np.math.factorial(n - x)) * p ** x * (1 - p) ** (n - x)

    @staticmethod
    def Geometric(x, p):
        """Geometric distribution"""
        return p * (1 - p) ** (x - 1)

    @staticmethod
    def Poisson(x, lam):
        """Poisson distribution"""
        return lam ** x * np.exp(-lam) / np.math.factorial(x)


# two-dimensional action space,n is the number of states
# if you want to change the distribution,
# you should change the step function and sample function
class Env(object):
    def __init__(self, n, mu=2, sigma=3, lam=1, alpha=2, beta=3, p=0.5):
        self.n = n
        self.action_space = np.arange(n)
        self.observation_space = np.arange(n)

        # define the probability distribution
        # continuous
        self.mu = mu
        self.sigma = sigma

        self.lam = lam

        self.alpha = alpha
        self.beta = beta
        # discrete
        self.p = p
        self.lam = lam

        # state
        self.state = 0

    @staticmethod
    def reward(action, target):
        """
        reward function
        :param action:  action
        :param target:  target
        :return:    reward
        """
        # reward
        return -np.abs(action - target)

    def step(self, action):
        """
        step function
        :param self:
        :param action:  action
        :return:    next state, reward
        """
        # distribution-----------------------------------------------change
        target = ProbabilityDensity.Normal(self.state, self.mu, self.sigma)

        # using action and target to calculate the reward
        r = self.reward(action, target)
        # sample the next state using the probability distribution
        self.sample()
        return self.state, r

    def reset(self):
        """
        reset function
        :param self:
        :return:    state
        """
        self.sample()
        return self.state

    def sample(self):
        """
        sample a state
        :return:
        """
        # distribution-----------------------------------------------change
        p_list = [ProbabilityDensity.Normal(i, self.mu, self.sigma) for i in range(self.n)]
        p_list = [p / sum(p_list) for p in p_list]
        self.state = np.random.choice(range(self.n), p=p_list)


def testProbabilityDensity():
    # test ProbabilityDensity function

    import matplotlib.pyplot as plt
    x = np.linspace(-5, 5, 100)
    y1 = ProbabilityDensity.StandardNormal(x)
    mu = 2
    sigma = 3
    y2 = ProbabilityDensity.Normal(x, mu, sigma)
    lam = 1
    y3 = ProbabilityDensity.Exponential(x, lam)
    alpha = 2
    beta = 3
    y4 = ProbabilityDensity.Beta(x, alpha, beta)
    plt.rcParams['font.size'] = 5
    plt.subplot(221)
    plt.plot(x, y1, 'r', linewidth=1, marker='o', markersize=2)
    plt.subplot(222)
    plt.title('Normal (mu={}, sigma={})'.format(mu, sigma))
    plt.plot(x, y2, 'g', linewidth=1, marker='o', markersize=2)
    plt.subplot(223)
    plt.title('Exponential (lam={})'.format(lam))
    plt.plot(x, y3, 'b', linewidth=1, marker='o', markersize=2)
    plt.subplot(224)
    plt.title('Beta (alpha={}, beta={})'.format(alpha, beta))
    plt.plot(x, y4, 'y', linewidth=1, marker='o', markersize=2)
    plt.show()


def testDistribution():
    # test Distribution function

    import matplotlib.pyplot as plt
    n = 500
    c = 20
    p = 0.4

    Bernoulli = []
    for i in range(n):
        [p0, p1] = ProbabilityMass.Bernoulli(0, p), ProbabilityMass.Bernoulli(1, p)
        x = np.random.choice([0, 1], p=[p0, p1])
        Bernoulli.append(x)
    Binomial = []
    for i in range(n):
        p_list = [ProbabilityMass.Binomial(i, c, p) for i in range(c + 1)]
        p_list = [p / sum(p_list) for p in p_list]
        x = np.random.choice(range(c + 1), p=p_list)
        Binomial.append(x)
    geometric = []
    for i in range(n):
        p_list = [ProbabilityMass.Geometric(i, p) for i in range(1, c + 1)]
        p_list = [p / sum(p_list) for p in p_list]
        x = np.random.choice(range(1, c + 1), p=p_list)
        geometric.append(x)
    poisson = []
    lam = 0.5
    for i in range(n):
        p_list = [ProbabilityMass.Poisson(i, lam) for i in range(1, c + 1)]
        p_list = [p / sum(p_list) for p in p_list]
        x = np.random.choice(range(1, c + 1), p=p_list)
        poisson.append(x)

    plt.rcParams['font.size'] = 5
    plt.subplot(221)
    plt.title('Bernoulli (p={})'.format(p))
    plt.bar([0, 1], [Bernoulli.count(0), Bernoulli.count(1)], width=0.9, color=['b', 'g'])
    plt.subplot(222)
    plt.title('Binomial (p={}, n={})'.format(p, c))
    plt.plot(range(1, c + 1), [Binomial.count(i) for i in range(1, c + 1)], 'r', linewidth=1, marker='o', markersize=2)
    plt.subplot(223)
    plt.title('Geometric (p={}, n={})'.format(p, c))
    plt.plot(range(1, c + 1), [geometric.count(i) for i in range(1, c + 1)], 'g', linewidth=1, marker='o', markersize=2)
    plt.subplot(224)
    plt.title('Poisson (lam={}, n={})'.format(lam, c))
    plt.plot(range(1, c + 1), [poisson.count(i) for i in range(1, c + 1)], 'b', linewidth=1, marker='o', markersize=2)
    plt.show()


def testEnv():
    n = 20
    env = Env(n,mu=10,sigma=1)
    states = []
    rewards = []
    for i in range(500):
        state, reward = env.step(0)
        states.append(state)
        rewards.append(reward)
    import matplotlib.pyplot as plt
    plt.subplot(211)
    plt.plot(range(1, n + 1), [states.count(i) for i in range(1, n + 1)], 'r', linewidth=1, marker='o', markersize=2)
    plt.subplot(212)
    plt.plot(range(500), rewards, 'g', linewidth=1, marker='o', markersize=2)
    plt.show()


if __name__ == '__main__':
    # testProbabilityDensity()
    # testDistribution()
    testEnv()
