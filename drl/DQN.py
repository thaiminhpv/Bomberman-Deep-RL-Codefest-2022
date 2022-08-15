import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 8, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2)
        self.head =

