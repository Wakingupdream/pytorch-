# -*- coding: utf-8 -*-
# !/usr/bin/env python
import torch
import torch.nn as nn


"""
Train a model to learn:
"hello"  -->  "ohlol"
"""


batch_size = 1
input_size = 4
hidden_size = 4
num_layers = 1
seq_len = 5

idx2char = ["e", "h", "l", "o"]
x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 2]
one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[i] for i in x_data]
inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)
labels = torch.LongTensor(y_data)


# Rnncell
class ModelRnnCell(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(ModelRnnCell, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.runcell = nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

    def forward(self, input, hidden):
        hidden = self.runcell(input, hidden)
        return hidden

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)


net = ModelRnnCell(input_size, hidden_size, batch_size)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

# for epoch in range(15):
#     loss = 0
#     optimizer.zero_grad()
#     hidden = net.init_hidden()
#     print("Predicted string:", end="")
#     for input, label in zip(inputs, labels):
#         hidden = net(input, hidden)
#         loss += criterion(hidden, label)
#         _, idx = hiar[idx.item()], end="")
#     loss.backward()dden.max(dim=1)
# #         print(idx2ch
#     optimizer.step()
#     print(", Epoch [%d/15] loss=%.4f" % (epoch + 1, loss.item()))


#Rnn
class ModelRnn(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
        super(ModelRnn, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)

    def forward(self, input):
        hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        out, _ = self.rnn(input, hidden)
        return out.view(-1, self.hidden_size)


net_rnn = ModelRnn(input_size, hidden_size, batch_size, num_layers)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net_rnn.parameters(), lr=0.15)

for epoch in range(200):
    optimizer.zero_grad()
    outputs = net_rnn(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()
    print("Predicted:", "".join([idx2char[x] for x in idx]), end="")
    print(", Epoch [%d/15] loss=%.4f" % (epoch + 1, loss.item()))
