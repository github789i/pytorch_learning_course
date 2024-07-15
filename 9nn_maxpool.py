import torch
import torchvision
from torch import nn

# 输入图像矩阵
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)

# 输入数据形状要求：(minibatch,in_channels,𝑖𝐻,𝑖𝑊)
input = torch.reshape(input, (-1, 1, 5, 5))
print(input.shape)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

tudui = Tudui()
# output = tudui(input)
# print(output)

# 定义一个totensor的转换
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# 下载pytorch自带的CIFAR10数据集
test_data = torchvision.datasets.CIFAR10(root="./dataset/CIFAR10", train=False, transform=dataset_transform,
                                         download=True)

# drop_last:当最后一组batch大小不足batch_size时，是否遗弃
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

writer = SummaryWriter("logs")

step = 0
for data in test_loader:
    img, target = data
    output = tudui(img)
    writer.add_images("input", img, step)
    writer.add_images("output", output, step)

    step += 1

writer.close()