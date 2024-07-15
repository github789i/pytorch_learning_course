import torch
# 加载模型方式1
import torchvision
from torch import nn

model1 = torch.load("model/vgg16_method1.pth")
print(model1)

# 加载模型方式2：加载字典形式保存的模型
vgg16 = torchvision.models.vgg16(pretrained=False)
model_para = torch.load("model/vgg16_method2.pth")
vgg16.load_state_dict(model_para)
print(vgg16)

# 陷阱1
# class Tudui(nn.Module):
#
#     def __init__(self) -> None:
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x

# 1、需重新写Tudui类
# 2、或者从其他文件中导入该类 from model_save import *
model = torch.load("model/tudui_method1.pth")
print(model)