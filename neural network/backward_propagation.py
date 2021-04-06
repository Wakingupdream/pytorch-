# -*- coding: utf-8 -*-
# !/usr/bin/env python

import torch

data_x = [1, 2, 3, 4]
data_y = [2, 4, 6, 8]


w = torch.Tensor([6])
w.requires_grad = True


def forward(x):
    return x * w


def loss(x, y):
    y_pre = forward(x)
    return (y - y_pre) ** 2


for epoch in range(200):
    for x, y in zip(data_x, data_y):
        l = loss(x, y)
        l.backward()
        w.data -= 0.001 * w.grad.item()
        w.grad.data.zero_()

print(forward(7).item())