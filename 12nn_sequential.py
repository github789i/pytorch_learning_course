import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
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
print(tudui)
# 生成测试数据（batch_size, channel, H, W）
input = torch.ones((128, 3, 32, 32))
output = tudui(input)
print(output.shape)

# 通过tensorboard进行神经网络结构可视化
writer = SummaryWriter("logs_seq")
writer.add_graph(tudui, input)
writer.close()