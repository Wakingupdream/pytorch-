# -*- coding: utf-8 -*-
# !/usr/bin/env python
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn


class DiabetesData(Dataset):
    def __init__(self):
        data_pd = pd.read_csv("../data/diabetes.csv")
        data_np = torch.Tensor(np.array(data_pd, dtype=np.float32))
        self.len = data_np.shape[0]
        self.data_x = data_np[:, :-1]
        self.data_y = data_np[:, [-1]]

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return self.len


dataset = DiabetesData()
train_loader = DataLoader(dataset, shuffle=True, batch_size=32)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(8, 50)
        self.linear2 = nn.Linear(50, 30)
        self.linear3 = nn.Linear(30, 10)
        self.linear4 = nn.Linear(10, 1)
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act(self.linear1(x))
        x = self.act(self.linear2(x))
        x = self.act(self.linear3(x))
        x = self.sigmoid(self.linear4(x))
        return x


model = Model()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epoch_loss = []
for epoch in range(3000):
    ls = 0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        pre = model(inputs)
        ls = criterion(pre, labels)
        optimizer.zero_grad()
        ls.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        epoch_loss.append(ls.item())
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.plot(range(len(epoch_loss)), epoch_loss)
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.show()
