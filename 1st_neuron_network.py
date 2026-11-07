import torch
import torch.nn as nn

n_in, n_h, n_outt, batch_size = 10,5,1,10

X=torch.rand(batch_size,n_in)
y = torch.tensor([[1.0], [0.0], [0.0],
                 [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])

model=nn.Sequential(
    nn.Linear(n_in,n_h),
    nn.ReLU(),
    nn.Linear(n_h,n_outt),
    nn.Sigmoid()
)

print(model)

criterion=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

losses = []
for epoch in range(100):
    y_pred=model(X)
    loss=criterion(y_pred,y)
    losses.append(loss.item())
    print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


from matplotlib import pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(range(1, 101), losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid()
plt.show()

y_pred_final = model(X).detach().numpy()
y_actual = y.numpy()

plt.figure(figsize=(8, 5))
plt.plot(range(1, batch_size + 1), y_actual, 'o-', label='Actual', color='blue')
plt.plot(range(1, batch_size + 1), y_pred_final, 'x--', label='Predicted', color='red')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.grid()
plt.show()



# another example
n_samples = 1000
data = torch.rand(n_samples, 2)*4-2
labels = (data[:, 0]**2 + data[:, 1]**2 < 1).float().unsqueeze(1) # 生成标签，圆内为1，圆外为0

plt.figure(figsize=(6, 6))
plt.scatter(data[:, 0], data[:, 1], c=labels.squeeze(), cmap='coolwarm')
plt.title('Generated Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.show()

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x=self.fc1(x)
        x=torch.relu(x)
        x=self.fc2(x)
        x=self.sigmoid(x)
        return x

model=SimpleNN()

criterion=nn.BCELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

epochs = 100
for epoch in range(epochs):
    outputs = model(data)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 可视化决策边界
def plot_decision_boundary(model, data):
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = torch.meshgrid(torch.arange(x_min, x_max, 0.1), torch.arange(y_min, y_max, 0.1), indexing='ij')
    grid = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1)
    predictions = model(grid).detach().numpy().reshape(xx.shape)
    plt.contourf(xx, yy, predictions, levels=[0, 0.5, 1], cmap='coolwarm', alpha=0.7)
    plt.scatter(data[:, 0], data[:, 1], c=labels.squeeze(), cmap='coolwarm', edgecolors='k')
    plt.title("Decision Boundary")
    plt.show()

plot_decision_boundary(model, data)