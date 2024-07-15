from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

image_path = "dataset/train/bees_image/16838648_415acd9e3f.jpg"
writer = SummaryWriter("logs")
img = Image.open(image_path)
img_array = np.array(img)

# tag:标题，img_tensor：图片数据，global_step：步长，dataformats：数据形状，H高度，W宽度，C通道
writer.add_image(tag="test", img_tensor=img_array, global_step=2, dataformats="HWC")

for i in range(100):
    # tag:标题
    # scalar_value:y轴值
    # global_step:x轴值
    writer.add_scalar(tag="y=2x", scalar_value=2*i, global_step=i)


writer.close()