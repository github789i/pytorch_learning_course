import torchvision

# 完整的模型训练套路
from torch import nn
from torch.utils.data import DataLoader

train_data = torchvision.datasets.CIFAR10(root="dataset/CIFAR10", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="dataset/CIFAR10", train=False, transform=torchvision.transforms.ToTensor(),
                                          download=True)

# 数据集大小
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"the length of train_data:{train_data_size}")
print(f"the length of test_data:{test_data_size}")

# 利用dataloader构造数据集
train_loader = DataLoader(train_data, batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)

# 搭建神经网络
class Tudui(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(

        )