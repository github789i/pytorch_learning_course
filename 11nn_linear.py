import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("dataset/CIFAR10", train=False, transform=torchvision.transforms.ToTensor(),
                                      download=True)

data_loader = DataLoader(dataset, batch_size=64, drop_last=True)

class Tudui(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output

tudui = Tudui()
for data in data_loader:
    img, target = data
    # 确定输入维度：torch.Size([196608])
    flatten = torch.flatten(img)
    print(flatten.shape)
    output = tudui(flatten)
    print(output.shape)
