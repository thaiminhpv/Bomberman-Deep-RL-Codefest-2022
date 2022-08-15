import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class DQN(nn.Module):
    def __init__(self, h, w, c, outputs):
        super(DQN, self).__init__()

        self.online = nn.Sequential(
            nn.Conv2d(c, 8, kernel_size=3, stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(5120, 256),
            nn.ReLU(),
            nn.Linear(256, outputs),
            nn.Softmax(1)
        )

        self.target = copy.deepcopy(self.online)

        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, x, model):
        x = x.permute(0, 3, 1, 2)
        if model == 'online':
            return self.online(x)
        elif model == 'target':
            return self.target(x)
