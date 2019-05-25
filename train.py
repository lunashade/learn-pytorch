#!/usr/bin/env python3

"""sample training code to learn PyTorch"""
import time

import torch
from torch import nn, optim

from .data_mnist import params, test_loader, train_loader
from .model import Net
from .utils import CSVWriter

# GPU setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)


# model and loss function
net = Net(**params)
net.to(device)
print(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)


# training loop
epochs = 3
t_start = time.time()
csvwriter = CSVWriter('loss.csv', 'epoch', 'iteration', 'loss', 'elapsed_time')
for epoch in range(epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print loss for each 100 iter.
        running_loss += loss.item()
        if i % 100 == 0:
            loss_to_log = running_loss/100
            elapsed_time = time.time() - t_start
            print(
                "epoch:", epoch,
                "iter:", i,
                "loss:", loss_to_log,
                "elapsed time:", elapsed_time,
            )
            csvwriter.write(epoch=epoch, iteration=i, loss=loss_to_log, elapsed_time=elapsed_time)
            running_loss = 0.0

print("Finish Training.")

# evaluation
correct = 0
total = 0

with torch.no_grad():
    for (inputs, labels) in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Accuracy[%]:", 100 * correct / total)
