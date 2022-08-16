import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class DQN(nn.Module):
    def __init__(self, h, w, c, outputs):
        super(DQN, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, outputs)
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        return self.body(x)
