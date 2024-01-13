import torch
import torch.nn as nn
import random
import struct

batch = random.randint(2, 4)
h = random.randint(2, 4)
w = 2
d = random.randint(2, 4)

x = torch.randn(batch, h, w, d, requires_grad=True)
weight = torch.randn(batch, h, 1, d, requires_grad=True)

x0, x1 = x.split(1, 2)
x0.requires_grad_()
x1.requires_grad_()

print("x0 requires grad")
print(x0.requires_grad)

y = x0.mul(weight).add(x1)

z = y.sum()

print(x)
print("-- split -- ")
print("x0")
print(x0)
print("x1")
print(x1)
print("x0.mul(weights).add(x1) = y")
print(y)

with open("x.bin", "wb") as file:
    file.write(struct.pack('i', 4))
    for s in x.shape: 
        file.write(struct.pack('i', s))
    for v in x.flatten().tolist():
        file.write(struct.pack('f', v))
    file.close()

with open("weight.bin", "wb") as file:
    file.write(struct.pack('i', 4))
    for s in weight.shape: 
        file.write(struct.pack('i', s))
    for v in weight.flatten().tolist():
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

#with open("x0_grad.bin", "wb") as file:
#    file.write(struct.pack('i', 4))
#    for s in x0.grad.shape: 
#        file.write(struct.pack('i', s))
#    for v in x0.grad.flatten().tolist():
#        file.write(struct.pack('f', v))
#    file.close()

#with open("x1_grad.bin", "wb") as file:
#    file.write(struct.pack('i', 4))
#    for s in x1.grad.shape: 
#        file.write(struct.pack('i', s))
#    for v in x1.grad.flatten().tolist():
#        file.write(struct.pack('f', v))
#    file.close()
