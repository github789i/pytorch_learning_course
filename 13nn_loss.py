import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

input = torch.tensor([1, 2, 3], dtype=torch.float32)
target = torch.tensor([1, 2, 5], dtype=torch.float32)

input = torch.reshape(input, (1, 1, 1, 3))
target = torch.reshape(target, (1, 1, 1, 3))

# L1 loss
loss = L1Loss(reduction="sum")
result = loss(input, target)

# MSE
loss_mse = MSELoss()
result_mse = loss_mse(input, target)

print(result)
print(result_mse)

# CrossEntropyLoss
x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1,3))
loss_cross = CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)