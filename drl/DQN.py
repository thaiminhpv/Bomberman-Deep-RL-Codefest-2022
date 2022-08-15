import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class DQN(nn.Module):
    def __init__(self, h, w, c, outputs):
        super(DQN, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(c, 8, kernel_size=7, stride=1, padding='same'),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding='same'),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(9216, 256),
            nn.ReLU(),
            nn.Linear(256, outputs),
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        return self.head(x)
