import torch
import random
import struct

batch = random.randint(2, 4)
m = random.randint(2, 16)
inner = random.randint(2, 16)
n = random.randint(2, 16)

a = torch.randn(batch, m, inner, requires_grad=True).abs() / 10
b = torch.randn(batch, inner, n, requires_grad=True).abs() / 10
a.view(batch, m, inner)
print(a.shape)
b.view(batch, inner, n)
print(b.shape)

c = a @ b

z = c.sum()

print(a)
print("@")
print(b)
print("=")
print(c)

with open("a.bin", "wb") as file:
    file.write(struct.pack('i', 3))
    for s in a.shape: 
        file.write(struct.pack('i', s))
    for v in a.flatten().tolist():
        file.write(struct.pack('f', v))
    file.close()

with open("b.bin", "wb") as file:
    file.write(struct.pack('i', 3))
    for s in b.shape: 
        file.write(struct.pack('i', s))
    for v in b.flatten().tolist():
        file.write(struct.pack('f', v))
    file.close()

with open("c.bin", "wb") as file:
    file.write(struct.pack('i', 3))
    for s in c.shape: 
        file.write(struct.pack('i', s))
    for v in c.flatten().tolist():
        file.write(struct.pack('f', v))
    file.close()

z.backward()

print("a_grad:")
print(a.grad)
print("b_grad")
print(b.grad)

#with open("a_grad.bin", "wb") as file:
#    file.write(struct.pack('i', 3))
#    for s in a.grad.shape: 
#        file.write(struct.pack('i', s))
#    for v in a.grad.flatten().tolist():
#        file.write(struct.pack('f', v))
#    file.close()

#with open("b_grad.bin", "wb") as file:
#    file.write(struct.pack('i', 3))
#    for s in b.grad.shape: 
#        file.write(struct.pack('i', s))
#    for v in b.grad.flatten().tolist():
#        file.write(struct.pack('f', v))
#    file.close()

#with open("c_grad.bin", "wb") as file:
#    file.write(struct.pack('i', 2))
#    for s in c.grad.shape: 
#        file.write(struct.pack('i', s))
#    for v in c.grad.flatten().tolist():
#        file.write(struct.pack('f', v))
#    file.close()
