import math
import random
import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from drl.DQN import DQN
from drl.ReplayMemory import *
from drl.preprocessing import *


def train_one_episode(data) -> str:
    processed_data = process_raw_input(data)

    return '1234bx'
