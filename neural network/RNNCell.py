# -*- coding: utf-8 -*-
# !/usr/bin/env python
import torch


batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2


cell = torch.nn.RNNCell(input_size, hidden_size)


dataset = torch.randn(seq_len, batch_size, input_size)
hidden = torch.zeros(batch_size, hidden_size)
for idx, data in enumerate(dataset):
    print("=" * 20, idx, "=" * 20)
    print("Input size:", data.shape)
    hidden = cell(data, hidden)
    print("=" * 20, idx, "=" * 20)
    print(hidden)
