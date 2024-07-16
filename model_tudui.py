# 搭建神经网络
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(64*4*4, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        output = self.model1(x)
        return output

if __name__ == '__main__':
    tudui = Tudui()
    input = torch.ones((64, 3, 32, 32))
    output = tudui(input)
    print(output)
    print(torch.argmax(output))
    print(output.shape)
