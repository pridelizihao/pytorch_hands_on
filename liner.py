import torch
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)

X=torch.randn(100,2)
true_w=torch.tensor([[2.0],[3.0]])
true_b=4.0

Y=X@true_w+true_b+0.1*torch.randn(100,1)

print(X[:5])
print(Y[:5])

import torch.nn as nn

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear=nn.Linear(2,1)

    def forward(self,x):
        return self.linear(x)

model=LinearRegressionModel()

criterion=nn.MSELoss()

optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

num_epochs=1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs=model(X)
    loss=criterion(outputs,Y)
    loss.backward()
    optimizer.step()

    if (epoch+1)%100==0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'Learned weights: {model.linear.weight.data.numpy()}')
print(f'Learned bias: {model.linear.bias.data.numpy()}')

with torch.no_grad():
    model.eval()
    predictions=model(X)

plt.scatter(X[:,0], Y, color='blue', label='Actual Data')
plt.scatter(X[:,0], predictions, color='red', label='Predicted Data')
plt.legend()
plt.title('Linear Regression Predictions')
plt.show()