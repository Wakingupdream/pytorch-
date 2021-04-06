# -*- coding: utf-8 -*-
# !/usr/bin/env python
from torch import optim
import torch.nn as nn
import torch


x_train = torch.Tensor([[1], [2], [3], [4]])
y_train = torch.Tensor([[2], [4], [6], [8]])


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()
optimizer = optim.SGD(model.parameters(), lr=1e-3)
ls = nn.MSELoss()
for epoch in range(1000):
    pre = model(x_train)
    loss = ls(pre, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print("loss is", loss.item())


data_test = torch.FloatTensor([[11], [22], [32], [12]])
print(model(data_test))
print("w", model.linear.weight.item())
print("b", model.linear.bias.item())
