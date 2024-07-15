import torchvision
from torch import nn

# 是否加载预训练好的模型权重
vgg_flase = torchvision.models.vgg16(pretrained=False)
vgg_true = torchvision.models.vgg16(pretrained=True)

torchvision.datasets.CIFAR10("dataset/CIFAR10", train=True, transform=torchvision.transforms.ToTensor(),
                             download=True)

# 在最后添加一层：将最后的1000类别分类器 -> 10类别分类器
vgg_true.add_module("add_linear", nn.Linear(1000, 10))
print(vgg_true)
# 添加到classifier中，而不是添加到外围
# vgg_true.classifier.add_module("add_linear", nn.Linear(1000, 10))
# print(vgg_true)

# 修改最后一层：将4096-》1000 变为 4096 -》10
vgg_flase.classifier[6] = nn.Linear(4096, 10)
print(vgg_flase)
