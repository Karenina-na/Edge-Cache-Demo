import torch
import torch.nn as nn
from Env.env import Env
from param import *
import numpy as np
from torch.nn import functional as F

env = Env()
s, info = env.reset()
n_state = s.shape[0]
n_action = env.action_space.actions_index_number


# 分类器
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(S_dim, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, n_action),
        )

    def forward(self, x):
        output = self.net(x)
        return output


# 构造数据集
data = []
label = []
s, _ = env.reset()
for i in range(1000):
    s, r, done, info, _ = env.step(0)
    index = np.argmax(s[:S_dim])
    data.append(s[:S_dim])
    label.append(index)
    if done:
        s, info = env.reset()
        break

data = torch.as_tensor(np.asarray(data), dtype=torch.float32)
label = torch.as_tensor(np.asarray(label), dtype=torch.float32)

# 预处理
data = F.normalize(data, p=2, dim=1)
label = torch.eye(n_action)[label.long()]

print(data.shape)
print(label.shape)

data_set = torch.utils.data.TensorDataset(data, label)
data_loader = torch.utils.data.DataLoader(data_set, batch_size=128, shuffle=False)

# 训练分类器
classifier = Classifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1)

for epoch in range(1000):
    for i, (data, label) in enumerate(data_loader):
        optimizer.zero_grad()
        output = classifier(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
    print('epoch:', epoch, 'loss:', loss.item())

# 测试分类器
s, info = env.reset()
cache_hit = []
cache_total = []
for i in range(10000):
    outputs = classifier(torch.as_tensor(np.array([s[:S_dim]]), dtype=torch.float32))
    probabilities = torch.softmax(outputs, dim=1)
    a = torch.argmax(probabilities, dim=1)
    s, r, done, info, _ = env.step(a[0])
    cache_hit.append(info['cache_hit'])
    cache_total.append(info['cache_total'])
    if done:
        s, info = env.reset()

print('-' * 100)
print('cache hit rate:', sum(cache_hit) / sum(cache_total))