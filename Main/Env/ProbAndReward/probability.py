import numpy as np


# Define the probability density function
class ProbabilityDensity(object):
    @staticmethod
    def StandardNormal(x):
        """Standard normal distribution"""
        return 1 / np.sqrt(2 * np.pi) * np.exp(-x ** 2 / 2)

    @staticmethod
    def Zipf(x, a, n):
        """

        :param x:
        :param a:
        :param n:
        :return:
        """
        return (1 / (x + 1) ** a) / np.sum(1 / (np.arange(1, n + 1) ** a))

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


if __name__ == '__main__':
    # testProbabilityDensity()
    testDistribution()