import torch
import torch.nn as nn
import random


class Linear(nn.Module):

    def __init__(self, in_features, out_features):
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features

        self.weight = torch.nn.Parameter(torch.rand(in_features, out_features) * (2/in_features) - 1/in_features)
        self.bias = torch.nn.Parameter(torch.randn(out_features) * (2/in_features) - 1/in_features)

    def forward(self, x):
        return (x @ self.weight) # + self.bias

B = random.randint(1,3)
m = random.randint(2,4)
in_features = random.randint(2,4)
out_features = random.randint(5,8)

ln_torch = nn.Linear(in_features, out_features)
ln_artisnal = Linear(in_features, out_features)

x = torch.randn(B, m, in_features, requires_grad=True)

#ln_artisnal.weight = ln_torch.weight
#ln_artisnal.bias = ln_torch.bias

ln_torch.weight = ln_artisnal.weight
ln_torch.bias = ln_artisnal.bias

y_torch = ln_torch(x)
y_artisnal = ln_artisnal(x)

print("torch result:")
print(y_torch)
print("artisnal results:")
print(y_artisnal)

error = y_torch.abs() - y_artisnal.abs()
print("error")
print(error)

mse = error.pow(2).mean()
print("mse")
print(mse)