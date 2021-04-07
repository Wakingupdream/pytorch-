# -*- coding: utf-8 -*-
# !/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def cross_entroy_loss_test():
    # 理解交叉熵损失
    criterion = nn.CrossEntropyLoss()
    y_real = torch.LongTensor([2, 0, 1])  # 共三类，第一个样本属于2，第二个样本属于1，……
    y_pre1 = torch.Tensor([[0.1, 0.2, 0.9],
                           [1.1, 0.1, 0.2],
                           [0.2, 2.1, 0.1]])
    y_pre2 = torch.Tensor([[0.8, 0.2, 0.3],
                           [0.2, 0.3, 0.5],
                           [0.2, 0.2, 0.5]])
    loss1 = criterion(y_pre1, y_real)
    loss2 = criterion(y_pre2, y_real)
    print("Batch loss1 =", loss1.item(), "Batch loss2 =", loss2.item())

    y_pre1_sf = F.softmax(y_pre1, dim=1)
    y_pre1_sf_lg = torch.log(y_pre1_sf)
    print(-(y_pre1_sf_lg[0][2] + y_pre1_sf_lg[1][0] + y_pre1_sf_lg[2][1]) / 3)


batch_size = 64
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,),
                                                     (0.3081,))])
train_dataset = datasets.MNIST(root="../data/", train=True, download=True,
                               transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root="../data/", train=False, download=True,
                              transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, 216)
        self.linear3 = nn.Linear(216, 128)
        self.linear4 = nn.Linear(128, 64)
        self.linear5 = nn.Linear(64, 10)
        self.act = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.act(self.linear1(x))
        x = self.act(self.linear2(x))
        x = self.act(self.linear3(x))
        x = self.act(self.linear4(x))
        x = self.linear5(x)
        return x


model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=model.parameters(), lr=0.01)


def train(epoch):
    running_loss = 0
    for i, data in enumerate(train_loader):
        inputs, target = data
        outputs = model(inputs)
        loss = criterion(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            print("[epoch {0:4d},iteration {1:4d}], running_loss is {2:f}".
                  format(epoch + 1, i + 1, running_loss / 300))
            running_loss = 0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = torch.max(model(inputs), dim=1)[1]
            correct += torch.sum(outputs == labels).item()
            total += labels.shape[0]
    print("Accuracy in test set is{0:f}%%".format(100 * correct / total))


for epoch in range(100):
    train(epoch)
    if (epoch + 1) % 20 == 0:
        test()
