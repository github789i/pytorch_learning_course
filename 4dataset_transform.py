import torchvision
from torch.utils.tensorboard import SummaryWriter

# 定义一个totensor的转换
dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# 下载pytorch自带的CIFAR10数据集
train_set = torchvision.datasets.CIFAR10(root="./dataset/CIFAR10", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset/CIFAR10", train=False, transform=dataset_transform, download=True)

print(test_set[0])
print(test_set.classes)
img, target = test_set[0]
print(img)
print(target)
print(test_set.classes[target])

# 查看图片
writer = SummaryWriter("logs")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)
writer.close()
