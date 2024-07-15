import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Tudui(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model1 = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        output = self.model1(x)
        return output

tudui = Tudui()
dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

test_data = torchvision.datasets.CIFAR10(root="./dataset/CIFAR10", train=False, transform=dataset_transform,
                                         download=True)

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
optim = torch.optim.SGD(tudui.parameters(), lr=0.1)
loss = nn.CrossEntropyLoss()
for epoch in range(20):
    running_loss = 0.0
    for data in test_loader:
        img, target = data
        output = tudui(img)
        result_loss = loss(output, target)
        # 每次将优化器中的梯度置为0，防止上一步的梯度造成误差
        optim.zero_grad()
        result_loss.backward()
        # 根据损失函数更新的梯度，优化器对参数进行调整
        optim.step()
        running_loss += result_loss
    print(running_loss)