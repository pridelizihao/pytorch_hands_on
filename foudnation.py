import torch

a=torch.zeros(2,3)
print(a)

b=torch.ones(2,3)
print(b)

c=torch.rand(2,3)
print(c)

import numpy as np
numpy_array=np.array([[1,2,3],[4,5,6]])
tensor_from_numpy=torch.from_numpy(numpy_array)
print(tensor_from_numpy)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
d=torch.randn(2,3,device=device,dtype=torch.float)
print(d)

e=torch.randn(2,3)
f=torch.randn(2,3)
print(e+f)

g=torch.randn(2,3)
print(g.t())
print(g.shape)


# 创建一个需要梯度的张量
tensor_requires_grad=torch.tensor([1.0], requires_grad=True)
# 操作一下
tensor_result=tensor_requires_grad*2

tensor_result.backward()
print(tensor_requires_grad.grad)

x=torch.randn(2,3,requires_grad=True)
print(x)

y=x+2
z=y*y*3
out=z.mean()

print("out:",out)

out.backward()
print("x.grad:", x.grad)


import torch.nn as nn
import torch.optim as optim

# 定义一个简单的全连接神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 2)  # 输入层到隐藏层
        self.fc2 = nn.Linear(2, 1)  # 隐藏层到输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 使用ReLU激活函数
        x = self.fc2(x)
        return x

model = SimpleNN()
print("model", model)


x=torch.randn(1,2)  

# 前向传播
output=model(x)
print(output)

# 定义损失函数
criterion = nn.MSELoss()  # 均方误差损失函数

target=torch.randn(1,1)  # 目标输出

# 计算损失
loss = criterion(output, target)
print("loss:", loss)


# 定义优化器（使用Adam优化器）
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器
loss.backward()  # 反向传播计算梯度
optimizer.step()  # 更新模型参数

# 训练模型的一般步骤
"""
1.数据准备
- 收集和处理数据，包括清洗、标准化和归一化
- 将数据分为训练集、验证集和测试集
2.定义模型
- 选择模型架构，例如决策树，神经网络
- 初始化模型参数（权重和偏置）
3.选择损失函数
- 根据任务类型（分类，回归）选择合适的损失函数
4.选择优化器
- 选择优化算法（如SGD，Adam）来更新模型参数
5.前向传播
- 在每次迭代中，输入数据根据模型传递，进行前向传播，计算模型的输出
6.计算损失
- 使用损失函数计算模型输出与真实标签之间的差异
7.反向传播
- 利用自动求导计算损失相对于模型参数的梯度
8.更新参数
- 使用优化器根据计算出的梯度更新模型参数
9.迭代优化
- 重复步骤5到8，直到在验证集的性能上不在提升或者达到预定的迭代次数
10.模型评估
- 在测试集上评估模型性能，计算准确率、精确率、召回率等指标
11.模型调优
- 根据模型在测试集上的表现进行调参，改变学习率，增加正则化
12.部署模型
- 将训练好的模型部署到生产环境中，用于实际的预测任务
"""

import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 2)  # 输入层到隐藏层
        self.fc2 = nn.Linear(2, 1)  # 隐藏层到输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 使用ReLU激活函数
        x = self.fc2(x)
        return x

model = SimpleNN()
criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

X=torch.randn(10, 2)  # 100个样本，每个样本有2个特征
Y=torch.randn(10, 1)  # 100个样本，每个样本有1个目标值

print("Training the model...")
for epoch in range(100):  # 训练100个epoch
    optimizer.zero_grad()  # 清零梯度
    outputs = model(X)  # 前向传播
    loss = criterion(outputs, Y)  # 计算损失
    loss.backward()  # 反向传播计算梯度
    optimizer.step()  # 更新模型参数

    if (epoch+1) % 10 == 0:  # 每10个epoch打印一次损失
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')