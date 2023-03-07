import array

import torch
import torch.multiprocessing as mp
import numpy as np
import os
from Agent import Net
from env import Env

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# A3C
UPDATE_GLOBAL_ITER = 10  # update global network every 5 episodes
GAMMA = 0.9  # reward discount
MAX_EP = 2000  # maximum episode
PARAMETER_NUM = mp.cpu_count()  # the number of parameters
ENTROPY = 0.001  # entropy coefficient
LEARNING_RATE = 0.1  # learning rate
BETAS = (0.92, 0.999)  # Adam optimizer parameters

# number of edge cache
n = 20

N_S = 1  # states dimension
N_A = 1  # actions dimension

# env
env = Env(n, mu=10, sigma=1)


class SharedAdam(torch.optim.Adam):
    """
    The following two functions are copied from the original pytorch source code.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


def v_wrap(array):
    """
    Wrap the numpy array into a torch tensor
    :return:
    """
    np_array = np.array(array, dtype=np.float32)
    return torch.from_numpy(np_array)


def record(global_ep, global_ep_r, ep_r, res_queue, name):
    """
    Record the result
    :param global_ep:   the global episode
    :param global_ep_r: the global episode reward
    :param ep_r:        the episode reward
    :param res_queue:   the result queue
    :param name:        the name of the process
    :return:            None
    """
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
    )


def push_and_pull_net(opt, lnet, gnet, s_, bs, ba, br, gamma):
    """
    Push the gradients to the global network and pull the parameters from the global network
    :param opt: the optimizer
    :param lnet:    the local network
    :param gnet:    the global network
    :param s_:  the next state
    :param bs:  the state
    :param ba:  the action
    :param br:  the reward
    :param gamma:   the discount factor
    :return:    None
    """
    # get the value of the next state, add the batch dimension
    v_s_ = lnet.forward(v_wrap([[s_]]))[-1].data.numpy()[0, 0]

    # calculate the advantage   R(t) + gamma * V(s_(t+1)) - V(s_t)
    buffer_v_target = []
    for r in br[::-1]:  # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    # calculate the loss
    bs = np.array(bs).reshape(-1, 1)
    ba = np.array(ba).reshape(-1, 1)
    buffer_v_target = np.array(buffer_v_target).reshape(-1, 1)

    loss = lnet.loss_func(v_wrap(bs), v_wrap(ba), v_wrap(buffer_v_target))

    # calculate local gradients
    opt.zero_grad()
    loss.backward()
    opt.step()

    # push the gradients to the global network
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp.grad = lp.grad

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, global_sam, res_queue, name):
        """
        :param gnet:            global network
        :param opt:             optimizer
        :param global_ep:       global episode
        :param global_ep_r:     global episode reward
        :param res_queue:       result queue
        :param name:            worker name
        """
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.glo_sam, self.res_queue = \
            global_ep, global_ep_r, global_sam, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A, entropy_beta=ENTROPY)  # local network

    def run(self):
        """
        run
        :return:    None
        """
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            while True:
                self.glo_sam[s] += 1
                # choose action
                a = self.lnet.sample(v_wrap([[s]]))
                # take action
                s_, r, done = env.step(a.detach().numpy()[0])
                ep_r += r
                buffer_a.append(a.detach().numpy()[0])
                buffer_s.append(s)
                buffer_r.append(r)
                # update global and assign to local net
                if total_step % UPDATE_GLOBAL_ITER == 0:  # update global and assign to local net
                    # sync
                    push_and_pull_net(self.opt, self.lnet, self.gnet, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1
        self.res_queue.put(None)

        print("Worker {} finished".format(self.name))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    gnet = Net(N_S, N_A, entropy_beta=0)  # global network
    start = []
    for i in range(n):
        a = gnet.sample(v_wrap([[i]]))
        start.append(a.detach().numpy()[0])

    gnet.share_memory()  # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=LEARNING_RATE, betas=BETAS)  # global optimizer

    # global_ep, global_ep_r, res_queue are used to record the result,
    # global_ep is the global episode, global_ep_r is the global episode reward,
    # res_queue is used to record the result
    global_ep, global_ep_r, global_sample, res_queue = \
        mp.Value('i', 0), mp.Value('d', 0.), mp.Array("i", n), mp.Queue()

    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, global_sample, res_queue, i) for i in range(PARAMETER_NUM)]

    # start training
    [w.start() for w in workers]

    # record episode reward to plot
    res = []
    global_sample = array.array('i', global_sample)

    # change structure of the result queue
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break

    # wait for all workers to finish
    [w.join() for w in workers]

    print("Finished")

    # show training result
    score = []
    for i in range(n):
        a = gnet.sample(v_wrap([[i]]))
        score.append(a.detach().numpy()[0])

    # plot the result
    import matplotlib.pyplot as plt

    # plot the result
    plt.rcParams['font.size'] = 16
    plt.figure(figsize=(20, 25))
    plt.subplot(5, 1, 1)
    plt.plot(start, linewidth=1, marker='o', markersize=2)
    plt.title("DRL initial distribution")

    plt.subplot(5, 1, 2)
    plt.plot(score, linewidth=1, marker='o', markersize=2)
    plt.title("DRL distribution")

    x, y = env.checkTheDistribution()
    plt.subplot(5, 1, 3)
    plt.plot(x, y, 'b', linewidth=1, marker='o', markersize=2)
    plt.title('Probability of the distribution')

    plt.subplot(5, 1, 4)
    plt.plot(range(len(global_sample)), global_sample.tolist())
    plt.title('env sample times')

    plt.subplot(5, 1, 5)
    plt.plot(range(len(res)), res)
    plt.ylabel('average ep reward')
    plt.xlabel('Step')
    plt.title('each step average reward')
    plt.show()
