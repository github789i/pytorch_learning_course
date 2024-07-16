import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential

# image_path = "./dataset/dog.png"
image_path = "./dataset/airplane.png"
image = Image.open(image_path)
print(image)
# png为四通道，调用下列函数保留颜色通道（若图片本来为三通道，经此操作不变）
image = image.convert("RGB")
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("./model/tudui_16_gpu.pth", map_location=torch.device('cuda:0'))
print(model)
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    image = image.to(device)
    output = model(image)
print(output)

print(output.argmax(1).item())
