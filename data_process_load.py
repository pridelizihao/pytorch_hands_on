import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        x=torch.tensor(self.X_data[idx], dtype=torch.float32)
        y=torch.tensor(self.y_data[idx], dtype=torch.float32)
        return x, y

X_data=[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]]
Y_data=[1,0,1,0,1]

dataset=MyDataset(X_data, Y_data)

dataloader=DataLoader(dataset, batch_size=2, shuffle=True)

for epoch in range(1):
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        print(f"Epoch {epoch+1}, Batch {batch_idx+1}")
        print(f"inputs: {inputs}")
        print(f"labels: {labels}")

import torchvision.transforms as transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 数据增强示例
# transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomRotation(degrees=30),
#     transforms.RandomResizedCrop(size=128),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

image=Image.open('dog.jpg')

image_tensor=transform(image)
print(f"Transformed image tensor shape: {image_tensor.shape}")

import torchvision.datasets as datasets

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

for inputs, labels in train_loader:
    print(f"Batch inputs shape: {inputs.shape}")
    print(f"Batch labels shape: {labels.shape}")

# from torch.utils.data import ConcatDataset

# # 假设 dataset1 和 dataset2 是两个 Dataset 对象
# combined_dataset = ConcatDataset([dataset1, dataset2])
# combined_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)