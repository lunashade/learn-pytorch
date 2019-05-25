#!/usr/bin/env python3

"""sample training code to learn PyTorch"""
import os
import time

import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms

# GPU setting
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

# dataset
train_dataset = torchvision.datasets.MNIST(
    'data/',
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)
test_dataset = torchvision.datasets.MNIST(
    'data/',
    train=False,
    transform=transforms.ToTensor(),
)

print("train dataset length:", len(train_dataset))
print("test dataset length:", len(test_dataset))

# data loaders
train_batchsize = 100
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=train_batchsize,
    shuffle=True,
    num_workers=2,
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=100,
    shuffle=False,
    num_workers=2,
)

# dataset specific parameters
H, W, C = 28, 28, 1
n_class = 10


# model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(H*W*C, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_class)

    def forward(self, x):
        x = x.view(-1, H*W*C)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
net.to(device)
print(net)


# loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)


# training loop
epochs = 3
t_start = time.time()
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
            print(
                "epoch:", epoch,
                "iter:", i,
                "loss:", running_loss/100,
                "elapsed time:", time.time() - t_start
            )
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
