import torch
import torch.nn as nn
import random
import struct

b = random.randint(2, 4)
w = random.randint(2, 16)
in_features = 32
out_features = 64

class Linear(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.rand(out_features, in_features).to("cuda") * (2/in_features) - 1/in_features)
        if(bias): self.bias = nn.Parameter(torch.randn(out_features).to("cuda") * (2/in_features) - 1/in_features)

    def forward(self, x):
        return (x @ self.weight.t()) + self.bias

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ln = nn.Linear(in_features, out_features, bias=True).to("cuda")

x = torch.randn(b, w, in_features, requires_grad=True).to("cuda")

y = ln.forward(x)

z = y.sum()

print(x)
print("ln(x)")
print(y)

with open("x.bin", "wb") as file:
    file.write(struct.pack('i', x.dim()))
    for s in x.shape: 
        file.write(struct.pack('i', s))
    for v in x.flatten().tolist():
        file.write(struct.pack('f', v))
    file.close()

with open("y.bin", "wb") as file:
    file.write(struct.pack('i', y.dim()))
    for s in y.shape: 
        file.write(struct.pack('i', s))
    for v in y.flatten().tolist():
        file.write(struct.pack('f', v))
    file.close()

with open("ln_weight.bin", "wb") as file:
    file.write(struct.pack('i', ln.weight.dim()))
    for s in ln.weight.shape: 
        file.write(struct.pack('i', s))
    for v in ln.weight.flatten().tolist():
        file.write(struct.pack('f', v))
    file.close()


with open("ln_bias.bin", "wb") as file:
    file.write(struct.pack('i', ln.bias.dim()))
    for s in ln.bias.shape: 
        file.write(struct.pack('i', s))
    for v in ln.bias.flatten().tolist():
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

print("ln_weight_grad")
print(ln.weight)

with open("ln_weight_grad.bin", "wb") as file:
    file.write(struct.pack('i', ln.weight.grad.dim()))
    for s in ln.weight.grad.shape: 
        file.write(struct.pack('i', s))
    for v in ln.weight.grad.flatten().tolist():
        file.write(struct.pack('f', v))
    file.close()

print("ln_bias_grad")
print(ln.bias)

with open("ln_bias_grad.bin", "wb") as file:
    file.write(struct.pack('i', ln.bias.grad.dim()))
    for s in ln.bias.grad.shape: 
        file.write(struct.pack('i', s))
    for v in ln.bias.grad.flatten().tolist():
        file.write(struct.pack('f', v))
    file.close()