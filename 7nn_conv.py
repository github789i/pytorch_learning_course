import torch
import torch.nn.functional as F

# 输入图像矩阵
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])
# 卷积核
kernal = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

# 输入数据形状要求：(minibatch,in_channels,𝑖𝐻,𝑖𝑊)
# in_channels：灰度图像通道是1维，彩色RGB通道是三维
input = torch.reshape(input, (1, 1, 5, 5))
kernal = torch.reshape(kernal, (1, 1, 3, 3))

print(input.shape)
print(kernal.shape)

# stride:步长，向右移动步数，一行遍历完后向下移动相同步数
# padding：将图像四周扩充像素数，默认扩充值为0
output = F.conv2d(input, kernal, stride=1)
print(output)

output2 = F.conv2d(input, kernal, stride=2)
print(output2)