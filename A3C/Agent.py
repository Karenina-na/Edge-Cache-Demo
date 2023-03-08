import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def set_init(layer):
    """
    initialize weights
    :param layer:   layer
    :return:    None
    """
    nn.init.normal_(layer.weight, mean=0., std=0.1)
    nn.init.constant_(layer.bias, 0.1)



class Net(nn.Module):
    def __init__(self, s_dim, a_dim, entropy_beta=0.005):
        super(Net, self).__init__()
        self.s_dim = s_dim  # state dimension
        self.a_dim = a_dim  # action dimension
        self.entropy_beta = entropy_beta

        # cli
        self.cli1 = nn.Linear(s_dim, 128)
        self.cli2 = nn.Linear(128, 128)

        # policy
        self.policy_tan = nn.Tanh()
        self.policy1 = nn.Linear(128, 256)
        self.policy2 = nn.Linear(256, 512)
        self.policy = nn.Linear(512, a_dim)
        # value
        self.value_tan = nn.Tanh()
        self.value1 = nn.Linear(128, 256)
        self.value2 = nn.Linear(256, 512)
        self.value = nn.Linear(512, 1)

        # mu sigma
        self.mu = nn.Linear(128, a_dim)
        self.sigma = nn.Linear(128, a_dim)

        # initialize weights
        set_init(self.cli1)
        set_init(self.cli2)
        set_init(self.policy1)
        set_init(self.policy2)
        set_init(self.policy)
        set_init(self.value1)
        set_init(self.value2)
        set_init(self.value)
        set_init(self.mu)
        set_init(self.sigma)

        self.distribution = torch.distributions.Normal  # action distribution

    def forward(self, x):
        self.train()

        # cli
        cli = self.cli1(x)
        cli = self.cli2(cli)

        # policy
        p = self.policy_tan(cli)
        p = self.policy1(p)
        p = self.policy2(p)
        p = self.policy_tan(p)
        actions = self.policy(p)

        # mu sigma
        mu = 2 * torch.tanh(self.mu(cli))
        sigma = F.softplus(self.sigma(cli)) + 0.00001  # avoid 0

        # value
        v = self.value_tan(cli)
        v = self.value1(v)
        v = self.value2(v)
        v = self.value_tan(v)
        values = self.value(v)

        return actions, values, mu, sigma

    def sample(self, state):
        """
        choose action
        :param state:   state
        :return:    action
        """
        self.eval()
        actions, _, _, _ = self.forward(state)
        return actions

    def loss_func(self, state, action, target):
        """
        loss function
        :param state:   state
        :param action:   action
        :param target:     target
        :return:    loss
        """
        self.train()
        # actions: the number of each action，values: the value of the state
        actions, values, mu, sigma = self.forward(state)

        # calculate TD error
        td = target - values
        # critic loss
        c_loss = td.pow(2)

        # use the distribution to calculate the loss
        dist = torch.distributions.Normal(mu, sigma)
        log_probs = dist.log_prob(action).sum(1, keepdim=True)
        entropy = dist.entropy()
        a_loss = -log_probs * td.detach()

        # compute entropy loss
        entropy_loss = -self.entropy_beta * entropy

        # total loss-mean
        total_loss = c_loss + a_loss + entropy_loss
        return total_loss.mean()


if __name__ == '__main__':
    s = torch.randint(0, 10, [5, 1], dtype=torch.float32)
    a = torch.rand(5, 1)
    n = Net(1, 1)
    logits, values, mu, sigma = n(s)
    print(logits.shape)
    print(values.shape)
    print(n.sample(s))
    print(n.loss_func(s, a, values))
