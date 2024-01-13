import torch
import torch.nn as nn
import random
import struct

torch.device("cuda")

batch = random.randint(2, 4)
h = random.randint(2, 4)
w = random.randint(2, 8)
d = random.randint(2, 8)

#ln = nn.LayerNorm((w, d))

print(f"norm shape: [{w, d}]")

x = torch.randn(batch, h, w, d, requires_grad=True)

y = torch.layer_norm(x, (w, d))

z = y.sum()

print(x)
print(". layernorm = ")
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

print(f"x std={x.std(2)}, inv_std={1.0/x.std(2)}")

print(f"x_grad: (sum={x.grad.sum()}, min={x.grad.min()}, avg={x.grad.mean()})")
print(x.grad)


with open("x_grad.bin", "wb") as file:
    file.write(struct.pack('i', 4))
    for s in x.grad.shape: 
        file.write(struct.pack('i', s))
    for v in x.grad.flatten().tolist():
        file.write(struct.pack('f', v))
    file.close()
