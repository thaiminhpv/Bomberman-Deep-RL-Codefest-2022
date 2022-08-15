import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class DQN(nn.Module):
    def __init__(self, h, w, c, outputs):
        super(DQN, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(c, 5, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(5),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(18, 24, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(24),
            nn.ReLU(),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(42, 80, kernel_size=5, stride=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(80),
            nn.ReLU(),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(80, 160, kernel_size=3, stride=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(160),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(640, 256),
            nn.ReLU(),
            nn.Linear(256, 48),
            nn.ReLU(),
            nn.Linear(48, outputs),
        )

    def forward(self, x):
        _a = x.permute(0, 3, 1, 2)
        a = self.block1(_a)
        b = self.block2(torch.cat([_a, a], 1))
        c = self.block3(torch.cat([_a, a, b], 1))
        d = self.block4(c)
        return self.head(d)
