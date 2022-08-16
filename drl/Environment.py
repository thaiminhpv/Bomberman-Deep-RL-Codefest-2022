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

from src.Hero import Hero


# class Singleton(type):
#     _instances = {}
#
#     def __call__(cls, *args, **kwargs):
#         if cls not in cls._instances:
#             cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
#         return cls._instances[cls]


class Environment:
    """
    An Singleton Environment that have tick(data) function call from producer, and have step() function call from consumer
    """
    instance = None

    @classmethod
    def get_player_id(cls):
        return Environment.instance.player_id

    def __init__(self, cv: Condition = None, hero: Hero = None, player_id: str = 'player1-xxx'):
        print('Environment.__init__, this should be called only once')
        Environment.instance = self
        self.QUEUE = deque()
        self.player_id = player_id
        self.condition = cv
        if hero is not None:
            self.hero = hero

    def __del__(self):
        print('Environment.__del__, this should be called only once')
        self.__clear()
        del self.hero

    def __move(self, action: int):
        print('Environment.__move, action:', action)
        print('Environment.__move, self.hero:', self.hero)
        print('Environment.__move, self.hero.player_id:', self.hero.player_id)
        print('Environment.__move, self.player_id:', self.player_id)
        print('Environment.__move, self.condition:', self.condition)
        print('Environment.__move, self.hero.running:', self.hero.running)
        self.hero.move(action)

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
        self.__move(action)
        with self.condition:
            while self.wait_for_tick():
                self.condition.wait()
            return self.pop()
