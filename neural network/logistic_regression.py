# -*- coding: utf-8 -*-
# !/usr/bin/env python


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        res = torch.sigmoid(self.linear(x))
        return res


x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])
model = LogisticRegression()
creterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
for epoch in range(1000):
    y_pre = model(x_data)
    ls = creterion(y_pre, y_data)
    print("loss", ls.item())
    optimizer.zero_grad()
    ls.backward()
    optimizer.step()
