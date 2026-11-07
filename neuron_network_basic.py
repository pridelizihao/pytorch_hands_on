import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN,self).__init__()
        self.fc1=nn.Linear(2,2)  # 输入层到隐藏层
        self.fc2=nn.Linear(2,1)  # 隐藏层到输出层

    def forward(self,x):
        x=torch.relu(self.fc1(x))  # 激活函数
        x=torch.sigmoid(self.fc2(x))  # 输出层使用sigmoid激活函数
        return x

model = SimpleNN()
print(model)

# 激活函数的使用
import torch.nn.functional as F
input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
output = F.relu(input_tensor)
output = F.sigmoid(input_tensor)
output = F.tanh(input_tensor)

# 损失函数
criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()

# 优化器
import torch.optim as optim
optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练过程
X=torch.rand(10,2)
Y=torch.rand(10,1)

print("training...")
for epoch in range(100):
    model.train()
    optimizer.zero_grad()  # 清空梯度
    output=model(X)  # 前向传播
    loss=criterion(output,Y)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    test_input = torch.tensor([[0.5, 0.5]])
    test_output = model(test_input)
    loss = criterion(test_output, torch.tensor([[0.5]]))
    print(f'Test Loss: {loss.item():.4f}')
    print("Test input:", test_input)
    print("Test output:", test_output)
