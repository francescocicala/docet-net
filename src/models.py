import torch
from torch import nn, optim
import torch.nn.functional as F


class Net():
    def __init__(self, arch):
        self.arch = arch
        self.W = None
        
    def forward(self, x):
        for i in range(len(self.arch) // 2 - 1):
            x = F.relu(F.linear(x, weight=self.W[2 * i].T, bias=self.W[2 * i + 1]))
        x = F.linear(x, weight=self.W[-2].T, bias=self.W[-1])
        return x
    
    def set_weights(self, W):
        self.W = W
        
    def __call__(self, x):
        return self.forward(x)



class NetCifar10():
    def __init__(self, arch):
        self.arch = arch
        self._initialize_weights()
        
    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        for i in range(len(self.arch) // 2 - 1):
            x = F.relu(F.linear(x, weight=self.W[2 * i].T, bias=self.W[2 * i + 1]))
        x = F.linear(x, weight=self.W[-2].T, bias=self.W[-1])
        return x
    
    def set_weights(self, W):
        self.W = W
        
    def _initialize_weights(self):
        weights = []
        for shape in self.arch:
            weights.append(torch.randn(shape))
        self.W = weights
        
    def __call__(self, x):
        return self.forward(x)