import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import struct
import math

B = random.randint(2, 4)
seq_length = 16
n_embed = 64
embed_dim = 32

emb = nn.Embedding(n_embed, embed_dim)
emb.to("cuda")

x = torch.randint(0, n_embed, (B, seq_length)).to("cuda")

y = emb(x)

z = y.sum()

print("x")
print(x)
print(". embedding = ")
print(y)

with open("x.bin", "wb") as file:
    file.write(struct.pack('i', x.dim()))
    for s in x.shape: 
        file.write(struct.pack('i', s))
    for v in x.flatten().tolist():
        file.write(struct.pack('i', v))
    file.close()

with open("emb_weight.bin", "wb") as file:
    file.write(struct.pack('i', emb.weight.dim()))
    for s in emb.weight.shape: 
        file.write(struct.pack('i', s))
    for v in emb.weight.flatten().tolist():
        file.write(struct.pack('f', v))
    file.close()

with open("y.bin", "wb") as file:
    file.write(struct.pack('i', y.dim()))
    for s in y.shape: 
        file.write(struct.pack('i', s))
    for v in y.flatten().tolist():
        file.write(struct.pack('f', v))
    file.close()

z.backward()

#with open("x_grad.bin", "wb") as file:
#    file.write(struct.pack('i', 4))
#    for s in x.grad.shape: 
#        file.write(struct.pack('i', s))
#    for v in x.grad.flatten().tolist():
#        file.write(struct.pack('f', v))
#    file.close()

print("emb_weight_grad")
print(emb.weight.grad)

with open("emb_weight_grad.bin", "wb") as file:
    file.write(struct.pack('i', emb.weight.grad.dim()))
    for s in emb.weight.grad.shape: 
        file.write(struct.pack('i', s))
    for v in emb.weight.grad.flatten().tolist():
        file.write(struct.pack('f', v))
    file.close()