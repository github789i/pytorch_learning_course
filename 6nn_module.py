import torch
from torch import nn


class Tudui(nn.Module):
    # overwrite:alt+insert
    def __init__(self) -> None:
        super().__init__()

    # overwrite
    def forward(self, input):
        output = input + 1
        return output

tudui = Tudui()
x = torch.tensor(1.0)
output = tudui(x)
print(output)