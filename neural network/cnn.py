# -*- coding: utf-8 -*-
# !/usr/bin/env python

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

from torch import optim
from torchvision import transforms, datasets


def conv_test():
    input = torch.Tensor([3, 4, 6, 5, 7,
                          2, 4, 6, 8, 2,
                          1, 6, 7, 8, 4,
                          9, 7, 4, 6, 2,
                          3, 7, 5, 4, 1]).view(1, 1, 5, 5)  # batch_size, channel, width, height
    kernel = torch.Tensor([1, 2, 3,
                           4, 5, 6,
                           7, 8, 9]).view(1, 1, 3, 3)
    conv = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)  # in_channel, out_channel, kernel_size
    conv.weight.data = kernel.data
    output = conv(input)
    print(output)
    print(output.shape)


# 手写数字识别
batch_size = 64
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,),
                                                     (0.3081,))])
train_dataset = datasets.MNIST(root="../data", train=True, download=True,
                               transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
test_dataset = datasets.MNIST(root="../data", train=False, download=True,
                              transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, drop_last=True)


class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.pooling = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, 10)

    def forward(self, x):  # batch_size, channel, width, height: 1 * 1 * 28 * 28
        # x = self.conv1(x)  # 1 * 10 * 24 * 24
        # x = self.pooling(x)  # 1 * 10 * 12 * 12
        # x = F.relu(x)
        # x = self.conv2(x)  # 1 * 20 * 8 * 8
        # x = self.pooling(x)  # 1 * 20 * 4 * 4
        # x = F.relu(x)
        # x = self.fc(x.view(1, -1))
        # 以上七行简约写法
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


model = Cnn()
# 如果要使用gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=model.parameters(), lr=0.01)


def train(epoch):
    running_loss = 0
    for i, data in enumerate(train_loader):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
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
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = torch.max(model(inputs), dim=1)[1]
            correct += torch.sum(outputs == labels).item()
            total += labels.shape[0]
    print("Accuracy in test set is{0:f}%%".format(100 * correct / total))


for epoch in range(100):
    train(epoch)
    if (epoch + 1) % 20 == 0:
        test()
