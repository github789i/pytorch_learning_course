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

loss = nn.CrossEntropyLoss()
for data in test_loader:
    img, target = data
    output = tudui(img)
    result_loss = loss(output, target)
    # 损失函数会更新梯度
    result_loss.backward()
    print("ok")