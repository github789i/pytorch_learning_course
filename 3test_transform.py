from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import cv2

# python的用法 -》tensor数据类型
# 通过 transforms.ToTensor去看两个问题
# 1.transforms该如何被使用
# 2.为什么要使用tensor数据类型

img_path = "dataset/train/ants_image/0013035.jpg"
img_path_abs = r"D:\PycharmProject\pytrochLeaning\dataset\train\ants_image\0013035.jpg"

# 读入图片
# PIL Image类型
img = Image.open(img_path)
print(type(img))
# numpy.ndarray类型
cv_img = cv2.imread(img_path)

writer = SummaryWriter("logs")

# 1.transforms该如何被使用
# ToTensor
# 创建具体的工具
tensor_trans = transforms.ToTensor()
# 使用创建的工具
tensor_img = tensor_trans(img)
# print(tensor_img)

writer.add_image("Tensor_img", tensor_img)

# Normalize
print(tensor_img[0][0][0])
# 计算公式：output[channel] = (input[channel] - mean[channel]) / std[channel]
trans_norm = transforms.Normalize(mean=[3, 1, 9], std=[2, 6, 7])

img_norm = trans_norm(tensor_img)
print(img_norm[0][0][0])

# 1表示在同标题下第二张处显示
writer.add_image("Norm_img", img_norm, 1)

# Resize
print(img.size)
# (h,w)缩放
trans_resize = transforms.Resize((512, 512))
# img PIL -> resize -> img_resize PIL
img_resize = trans_resize(img)
print(img_resize)
# img_resize PIL -> totensor -> img_resize tensor
img_resize = tensor_trans(img_resize)
# print(img_resize)

writer.add_image("Resize", img_resize, 0)

# Compose - resize -2
# int 将更小的边进行缩放
trans_resize_2 = transforms.Resize(512)
# compose:按顺序执行
# 1.trans_resize_2 改变PIL image大小；
# 2.tensor_trans将PIL image 类型转化为tensor类型给writer显示
trans_compose = transforms.Compose([trans_resize_2, tensor_trans])
img_resize_2 = trans_compose(img)

writer.add_image("Resize", img_resize_2, 1)

# RandomCrop:随机裁剪
trans_random = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_random, tensor_trans])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)

writer.close()
# 在terminal中输入tensorboard --logdir=logs查看结果