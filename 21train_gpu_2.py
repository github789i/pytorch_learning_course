import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from model_tudui import *
from torch import nn
from torch.utils.data import DataLoader
import time

# 定义训练的设备
# device = torch.device("cpu")
# device = torch.device("cuda:0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 完整的模型训练套路
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

# 创建网络模型：从mdoel.py文件中导入设置好的模型，便于管理
tudui = Tudui()
tudui = tudui.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 训练次数
total_train_step = 0
# 测试次数
total_test_step = 0
# 训练轮数
epoch = 30

# 添加tensorboard可视化
writer = SummaryWriter("./logs_train")
start_time = time.time()
for i in range(epoch):
    print(f"---------第{i+1}轮训练开始-----------")

    # 开始训练
    tudui.train() #控制某些层
    for data in train_loader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(f"训练次数：{total_train_step}，花费时间：{end_time-start_time}，loss:{loss.item()}")
            writer.add_scalar("train_loos", loss.item(), total_train_step)

    # 测试开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    # 测试不需要对梯度进行调整
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print(f"整体测试集上的Loss：{total_test_loss}")
    print(f"整体测试集上的Accuracy：{total_accuracy/test_data_size}")
    writer.add_scalar("test_loos", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    # 模型保存
    torch.save(tudui, f"./model/tudui_{i}_gpu.pth")
    # torch,save(tudui.state_dict(), f"./model/tudui_{i}.pth")
    print("模型已保存")

writer.close()