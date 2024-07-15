import torchvision
from torch.utils.data import DataLoader

# 定义一个totensor的转换
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# 下载pytorch自带的CIFAR10数据集
test_data = torchvision.datasets.CIFAR10(root="./dataset/CIFAR10", train=False, transform=dataset_transform, download=True)

# drop_last:当最后一组batch大小不足batch_size时，是否遗弃
test_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

# 获取第一张图片
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("logs")
step = 0
for i in test_loader:
    img, target = i
    writer.add_images("test_data", img, step)
    step += 1

writer.close()
