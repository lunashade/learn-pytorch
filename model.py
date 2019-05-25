"""pytorch model"""
from torch import nn
from torch.nn import functional as F


# model
class Net(nn.Module):
    def __init__(self, **params):
        """params needs H, W, C, n_class."""
        super(Net, self).__init__()
        for key in params:
            setattr(self, key, params[key])

        self.conv1 = nn.Conv2d(self.C, 128, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.fc1 = nn.Linear(self.H//4 * self.W//4 * 128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.n_class)

    def forward(self, x):
        # H x W
        x = F.relu(self.conv1(x))
        # H/2 x W/2
        x = F.relu(self.conv2(x))
        # H/4 x W/4
        x = x.view(-1, self.H//4 * self.W//4 * 128)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
