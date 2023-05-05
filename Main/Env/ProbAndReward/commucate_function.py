import numpy as np
import math


def ground_communicate():
    # 节点跟基站通信
    cfg_ground = {
        "d": 1000,  # 节点到基站的距离
        "d0": 10,  # 参考路径 m
        "lamda": 0.125,  # 波长 m
        "n": 2,  # 路径损耗系数
        "Bg": 1000000,  # 基站的带宽 Hz
        "Pg": 1,  # 信号传输功率 w
        "sigma": 0.1,  # 信道内高斯噪声功率 db
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
        "d": 3000,  # 节点到基站的距离 m
        "dH": 20,  # 高度差 m
        "n": 4,  # 路径损耗系数
        "Bag": 1000000,  # 信道带宽 Hz
        "Pag": 0.01,  # 信号传输功率 w
        "sigma": 10e-10,  # 信道内高斯噪声功率 dbm
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
        "Gt": 17,  # 发射天线增益 db
        "Gr": 25,  # 接收天线增益 db
        "lamda": 0.02,  # 波长 m
        "d": 800000,  # 节点到基站的距离 m
        "Bsa": 2400000,  # 信道带宽 Hz
        "Psa": 200,  # 信号传输功率 w
        "a": 3,  # 路径损耗系数
        "sigma": 10000,  # 信道内高斯噪声功率 db
    }

    # 路径损耗模型
    def path_loss_model(d, Gt, Gr, lamda):
        # PL(d) = -10 * log10((Gt * Gr * lamda^2) / (16 * pi^2 * d^2))
        """
        :param d: 传播距离
        """
        return -10 * math.log10((Gt * Gr * lamda ** 2) / (16 * math.pi ** 2 * d ** 2))

    # Csa = Bsa * log2(1 + (Psa * path_loss_model(d, Gt, Gr, lamda) * exp(-a / 10)) / sigma)
    Csa = cfg_sa["Bsa"] * \
          math.log2(1 + (cfg_sa["Psa"] * path_loss_model(cfg_sa["d"], cfg_sa["Gt"], cfg_sa["Gr"], cfg_sa["lamda"])
                         * math.exp(-cfg_sa["a"] / 10)) / cfg_sa["sigma"])
    return Csa


file_weight = {
    i: 1000 for i in range(0, 10000)
}


# 计算传输时间
def Calculate_time(index):
    rate = ground_communicate()  # 地面
    # rate = plane_communicate()    # 无人机
    # rate = satellite_communicate()    # 卫星
    time = file_weight[index] / rate
    return time


if __name__ == "__main__":
    print(ground_communicate())
    print(plane_communicate())
    print(satellite_communicate())
