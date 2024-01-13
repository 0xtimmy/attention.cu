import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import struct
import math

torch.device("cuda")

batch = 1
block_size = 16
n_embed = 32
n_head = 4

class SelfAttention(nn.Module):
    
    def __init__(self, n_embd, n_head: int):
        nn.Module.__init__(self)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

        self.n_head = n_head
        self.n_embd = n_embd

        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape

        proj_in = self.c_attn(x)
        q, k, v  = proj_in.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, math.floor(self.n_embd / self.n_head)).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, math.floor(self.n_embd / self.n_head)).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, math.floor(self.n_embd / self.n_head)).transpose(1, 2) # (B, nh, T, hs)

        k_transpose = k.transpose(-2, -1)
        att = (q @ k_transpose) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.c_proj(y)

        return y


sa = SelfAttention(n_embed, n_head)
sa.to("cuda")

x = torch.randn(batch, block_size, n_embed, requires_grad=True).to("cuda")

y = sa(x)

z = y.sum()

print("x")
print(x)
print(". self attention = ")
print(y)

with open("x.bin", "wb") as file:
    file.write(struct.pack('i', x.dim()))
    for s in x.shape: 
        file.write(struct.pack('i', s))
    for v in x.flatten().tolist():
        file.write(struct.pack('f', v))
    file.close()

with open("c_attn_weight.bin", "wb") as file:
    file.write(struct.pack('i', sa.c_attn.weight.dim()))
    for s in sa.c_attn.weight.shape: 
        file.write(struct.pack('i', s))
    for v in sa.c_attn.weight.flatten().tolist():
        file.write(struct.pack('f', v))
    file.close()

with open("c_attn_bias.bin", "wb") as file:
    file.write(struct.pack('i', sa.c_attn.bias.dim()))
    for s in sa.c_attn.bias.shape: 
        file.write(struct.pack('i', s))
    for v in sa.c_attn.bias.flatten().tolist():
        file.write(struct.pack('f', v))
    file.close()

with open("c_proj_weight.bin", "wb") as file:
    file.write(struct.pack('i', sa.c_proj.weight.dim()))
    for s in sa.c_proj.weight.shape: 
        file.write(struct.pack('i', s))
    for v in sa.c_proj.weight.flatten().tolist():
        file.write(struct.pack('f', v))
    file.close()

with open("c_proj_bias.bin", "wb") as file:
    file.write(struct.pack('i', sa.c_proj.bias.dim()))
    for s in sa.c_proj.bias.shape: 
        file.write(struct.pack('i', s))
    for v in sa.c_proj.bias.flatten().tolist():
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

print("c_attn_weight_grad")
print(sa.c_attn.weight.grad)

with open("c_attn_weight_grad.bin", "wb") as file:
    file.write(struct.pack('i', sa.c_attn.weight.grad.dim()))
    for s in sa.c_attn.weight.grad.shape: 
        file.write(struct.pack('i', s))
    for v in sa.c_attn.weight.grad.flatten().tolist():
        file.write(struct.pack('f', v))
    file.close()

print("c_attn_bias_grad")
print(sa.c_attn.bias.grad)

with open("c_attn_bias_grad.bin", "wb") as file:
    file.write(struct.pack('i', sa.c_attn.bias.grad.dim()))
    for s in sa.c_attn.bias.grad.shape: 
        file.write(struct.pack('i', s))
    for v in sa.c_attn.bias.grad.flatten().tolist():
        file.write(struct.pack('f', v))
    file.close()

print("c_proj_weight_grad")
print(sa.c_proj.weight.grad)

with open("c_proj_weight_grad.bin", "wb") as file:
    file.write(struct.pack('i', sa.c_proj.weight.grad.dim()))
    for s in sa.c_proj.weight.grad.shape: 
        file.write(struct.pack('i', s))
    for v in sa.c_proj.weight.grad.flatten().tolist():
        file.write(struct.pack('f', v))
    file.close()

print("c_proj_bias_grad")
print(sa.c_proj.bias.grad)

with open("c_proj_bias_grad.bin", "wb") as file:
    file.write(struct.pack('i', sa.c_proj.bias.grad.dim()))
    for s in sa.c_proj.bias.grad.shape: 
        file.write(struct.pack('i', s))
    for v in sa.c_proj.bias.grad.flatten().tolist():
        file.write(struct.pack('f', v))
    file.close()