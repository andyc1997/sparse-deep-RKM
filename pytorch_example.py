import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

# Simple example of pytorch optimization
# https://www.cl.cam.ac.uk/teaching/2021/LE49/probnn/3-3.pdf
class StraightLine(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return self.alpha + self.beta * x


class Y(nn.Module):
    def __init__(self):
        super(Y, self).__init__()
        self.f = StraightLine()
        self.sigma = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, y):
        pred = self.f(x)
        return -0.5 * torch.log(2 * np.pi * self.sigma ** 2) - (y - pred) ** 2 / (2 * self.sigma ** 2)


model = Y()
optimizer = optim.Adam(model.parameters(), lr = 1)
scheduler = ExponentialLR(optimizer, gamma = 0.1)

torch.manual_seed(0)
x = torch.tensor(np.arange(0, 100), dtype = torch.float32)
y = 2 + 5 * x + 0.5 * torch.randn((100,))

# standard regression
print(np.polyfit(x.numpy(), y.numpy(), 1))

# gradient descent
epoch = 0
n_epochs = 1000

while epoch <= n_epochs:
    optimizer.zero_grad()
    loglik = model(x, y)

    e = -torch.mean(loglik)
    e.backward() # get gradients w.r.t to parameters
    optimizer.step() # update parameters

    if epoch % 500 == 0:
        scheduler.step()  # update parameters

    if epoch % 100 == 0:
        print(f'epoch = {epoch} loglik={-e.item(): .6}')
    epoch += 1

# show parameter
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)