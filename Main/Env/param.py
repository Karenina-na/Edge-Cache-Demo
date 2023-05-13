# 缓存公共参数

A_dim = 10  # 缓存内容索引大小
S_dim = 10  # 状态空间
A_number = 4  # 缓存空间大小
Request_number = 100  # 一次请求的请求数量
Stop_number = 1000  # 环境请求最大数量

# 通信类型 ground plane satellite
communicate_type = "ground"
# communicate_type = "plane"
# communicate_type = "satellite"

# 通信超时时间
time_out = 130
# time_out = 230
# time_out = 340

a = 0.6  # zipf分布参数
w = 0  # 奖励比例系数 w越大缓存命中率考虑越多
