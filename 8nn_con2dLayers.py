import torch
import torchvision
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch import nn

# 定义一个totensor的转换
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# 下载pytorch自带的CIFAR10数据集
test_data = torchvision.datasets.CIFAR10(root="./dataset/CIFAR10", train=False, transform=dataset_transform,
                                         download=True)

# drop_last:当最后一组batch大小不足batch_size时，是否遗弃
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)


class Tudui(nn.Module):
    # overwrite:alt+insert
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    # overwrite
    def forward(self, x):
        x = self.conv1(x)
        return x


tudui = Tudui()

writer = SummaryWriter("logs")

step = 0
for data in test_loader:
    img, target = data
    output = tudui(img)
    # torch.Size([64, 3, 32, 32])
    # print(img.shape)
    # torch.Size([64, 6, 30, 30])
    # print(output.shape)
    writer.add_images("input", img, step)
    # 6通道无法显示，改为3通道
    # torch.Size([64, 6, 30, 30]) ->[xxx, 3, 30, 30]
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)

    step += 1

writer.close()
