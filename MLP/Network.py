import torch
from torch import nn
from torch.nn import functional as F

class Net(nn.Module):
    # Assign Intrinsic Properties of Your Neural Network
    def __init__(self):
        super().__init__()
        # Weights of Layer 1th and Layer 2th Are Intrinsic Properties
        self.fc1 = nn.Linear(784, 256, bias=True)
        self.fc2 = nn.Linear(256, 100, bias=True)
        self.fc3 = nn.Linear(100, 10, bias=True)

    # Wiring of Your Network
    def feed_forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu_(x)
        x = self.fc2(x)
        x = F.relu_(x)
        x = self.fc3(x)
        return x