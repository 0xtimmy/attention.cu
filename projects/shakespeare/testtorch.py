import torch

a = torch.ones(3, 32, 12)
b = torch.ones(32, 12)

c = a + b

print(c.shape)