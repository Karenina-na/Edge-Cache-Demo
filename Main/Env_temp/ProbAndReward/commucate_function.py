import numpy as np
import math
from Main.Env.ProbAndReward.probability import ProbabilityDensity
from Main.Env.param import *
import matplotlib.pyplot as plt


def ground_communicate():
    # 节点跟基站通信
    cfg_ground = {
        "d": 1000,  # 节点到基站的距离
        "d0": 10,  # 参考路径 m
        "lamda": 0.125,  # 波长 m
        "n": 2,  # 路径损耗系数
        "Bg": 1000000,  # 基站的带宽 Hz
        "Pg": 0.5,  # 信号传输功率 w
        "sigma": 0.01,  # 信道内高斯噪声功率 W
    }

    # 路径损耗模型
    def path_loss_model(d, d0, lamda, n):
        # PL(d) = 20 * log10((4 * pi * d0) / lamda) + 10 * n * log10(d0 / d)
        """
        :param d: 传播距离
        """
        return 20 * math.log10((4 * math.pi * d0) / lamda) + 10 * n * math.log10(d0 / d)

    # cg = Bg * log2(1 + (Pg * path_loss_model(d, d0, lamda, n)) / sigma)
    Cg = cfg_ground["Bg"] * \
         math.log2(1 + (cfg_ground["Pg"] *
                        path_loss_model(cfg_ground["d"], cfg_ground["d0"], cfg_ground["lamda"], cfg_ground["n"]))
                   / cfg_ground["sigma"])
    return Cg


def plane_communicate():
    # 节点跟无人机通信
    cfg_plane = {
        "d": 300,  # 节点到基站的距离 m
        "dH": 15,  # 高度差 m
        "n": 2,  # 路径损耗系数
        "Bag": 6000000,  # 信道带宽 Hz
        "Pag": 2,  # 信号传输功率 w
        "sigma": 0.01,  # 信道内高斯噪声功率 W
    }

    # 路径损耗模型
    def path_loss_model(d, dH, n):
        # PL(d) = PL(d0) + 10 * n * log10(d / dH)
        """
        :param d: 传播距离
        """
        return path_loss_model(dH, dH, n) + 10 * n * math.log10(d / dH)

    # cag = Bag * log2(1 + Pag/(sigma * dH ^ n))
    Cag = cfg_plane["Bag"] * \
          math.log2(1 + cfg_plane["Pag"] / (cfg_plane["sigma"] * cfg_plane["dH"] ** cfg_plane["n"]))
    return Cag


def satellite_communicate():
    # 节点跟卫星通信
    cfg_sa = {
        "Gt": 17,  # 发射天线增益 dbi
        "Gr": 25,  # 接收天线增益 dbi
        "lamda": 0.2,  # 波长 m
        "d": 1000000,  # 节点到基站的距离 m
        "Bsa": 10000000,  # 信道带宽 Hz
        "Psa": 150,  # 信号传输功率 w
        "a": 5,  # 路径损耗系数
        "sigma": 3,  # 信道内高斯噪声功率 W
    }

    # 路径损耗模型
    def path_loss_model(d, Gt, Gr, lamda):
        # PL(d) = -10 * log10((Gt * Gr * lamda^2) / (16 * pi^2 * d^2))
        """
        :param d: 传播距离
        """
        return -10 * math.log10((Gt * Gr * lamda ** 2) / (16 * math.pi ** 2 * d ** 2))

    # Csa = Bsa * log2(1 + (Psa * path_loss_model(d, Gt, Gr, lamda)) / sigma * exp(2a))
    Csa = cfg_sa["Bsa"] * \
          math.log2(1 + (cfg_sa["Psa"] * path_loss_model(cfg_sa["d"], cfg_sa["Gt"], cfg_sa["Gr"], cfg_sa["lamda"])
                         ) / (cfg_sa["sigma"] * math.exp(2 * cfg_sa["a"])))
    return Csa


file_weight = {
    i: ProbabilityDensity.Normal(i, A_dim / 2, 5) for i in range(0, A_dim)
}


# 计算传输时间
def Calculate_time(index):
    if communicate_type == "ground":
        rate = ground_communicate()  # 地面
    elif communicate_type == "plane":
        rate = plane_communicate()  # 无人机
    else:
        rate = satellite_communicate()  # 卫星
    file_scale = [1000000, 5000000]  # 文件大小范围
    file = (file_scale[1] - file_scale[0]) * file_weight[index] + file_scale[0]
    time = file / rate
    return time * 1000  # ms


if __name__ == "__main__":
    # print(str(ground_communicate()/1000000) + " Mbps")
    # print(str(plane_communicate()/1000000) + " Mbps")
    # print(str(satellite_communicate()/1000000) + " Mbps")
    time = []
    for i in range(10):
        print(Calculate_time(i))
        time.append(Calculate_time(i))
    plt.scatter(range(10), time)
    plt.show()