import torch
import torch.nn as nn
import random
import struct

torch.device("cuda")

batch = random.randint(2, 4)
h = random.randint(2, 4)
w = random.randint(2, 8)
d = random.randint(2, 8)

ge = nn.GELU()

x = torch.randn(batch, h, w, d, requires_grad=True)

y = ge(x)

z = y.sum()

print(x)
print(". gelu = ")
print(y)

with open("x.bin", "wb") as file:
    file.write(struct.pack('i', 4))
    for s in x.shape: 
        file.write(struct.pack('i', s))
    for v in x.flatten().tolist():
        file.write(struct.pack('f', v))
    file.close()

with open("y.bin", "wb") as file:
    file.write(struct.pack('i', 4))
    for s in y.shape: 
        file.write(struct.pack('i', s))
    for v in y.flatten().tolist():
        file.write(struct.pack('f', v))
    file.close()

z.backward()

print("x_grad:")
print(x.grad)


with open("x_grad.bin", "wb") as file:
    file.write(struct.pack('i', 4))
    for s in x.grad.shape: 
        file.write(struct.pack('i', s))
    for v in x.grad.flatten().tolist():
        file.write(struct.pack('f', v))
    file.close()
