"""MNIST dataset"""

import torch
import torchvision
from torchvision import transforms

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
params = {
    'H': 28,
    'W': 28,
    'C': 1,
    'n_class': 10,
}
