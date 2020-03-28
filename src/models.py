import torch
from torch import nn, optim
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hid = nn.Linear(2, 2)
        self.out = nn.Linear(2, 1)
    
    def forward(self, x):
        x = F.relu(self.hid(x))
        x = self.out(x)
        return x


class Teacher(nn.Module):
    def __init__(self, cardinality):
        super(Teacher, self).__init__()
        self.hid1 = nn.Linear(cardinality, 100)
        self.hid2 = nn.Linear(100, 100)
        self.out = nn.Linear(100, cardinality)
    
    def forward(self, x):
        x = F.relu(self.hid1(x))
        x = F.relu(self.hid2(x))
        x = self.out(x)
        return x