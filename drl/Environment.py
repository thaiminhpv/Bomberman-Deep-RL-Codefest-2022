import random
import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
from threading import Condition
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from drl.DQN import DQN
# import asyncio
from collections import deque


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Environment(metaclass=Singleton):
    """
    An Singleton Environment that have tick(data) function call from producer, and have step() function call from consumer
    """

    def __init__(self, cv: Condition = None, move=lambda x: print('move function not defined'), player_id: str = 'player1-xxx'):
        self.QUEUE = deque()
        self.condition = cv
        self.player_id = player_id
        if move is not None:
            def _move(action: int):
                move(action)
            self.move = _move

    def __clear(self):
        self.QUEUE.clear()

    def __push(self, item):
        self.__clear()
        self.QUEUE.append(item)

    def __set_condition(self, cv: Condition):
        self.condition = cv

    def __set_move_function(self, move):
        self.move = move

    def wait_for_tick(self):
        return not(bool(self.QUEUE))

    def pop(self):
        return self.QUEUE.popleft()

    def tick(self, data):
        self.__push(data)

    def step(self, action: int):
        self.__clear()
        self.move(action)
        with self.condition:
            while self.wait_for_tick():
                self.condition.wait()
            return self.pop()