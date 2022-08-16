import math
import os
import time
# import asyncio
import random
from threading import Condition

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
from drl.Environment import *
from drl.util import *

EVAL_MODE = False
resume = False

RANDOM_SEED = 420

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 600
TARGET_UPDATE = 500

if EVAL_MODE:
    resume = True
    EPS_START = 0.05

screen_height = 14
screen_width = 26
depth = 13
n_actions = 6

steps_done = 0
episode_durations = []

seed_everything(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN(screen_height, screen_width, depth, n_actions).to(device)
target_net = DQN(screen_height, screen_width, depth, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

if resume:
    if os.path.isfile('./model/dqn.pth'):
        print('load model')
        policy_net.load_state_dict(torch.load('./model/dqn.pth'))
        target_net.load_state_dict(torch.load('./model/dqn.pth'))
    else:
        print('no model')

optimizer = optim.Adam(policy_net.parameters(), lr=0.0001)
memory = ReplayMemory(10000)
criterion = nn.SmoothL1Loss()

mapping = {0: '←', 1: '→', 2: '↑', 3: '↓', 4: 'bomb', 5: 'stop'}

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward
            temp = policy_net(state[None, ...])
            confident = F.softmax(temp, dim=1).max(1).values.detach().item()
            print(f'{mapping[temp.argmax().item()]} : {confident * 100:.3f}%', end='\t')
            return temp.argmax().view(1, 1)
            # return policy_net(state[None, ...]).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        print()
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.stack(batch.state)
    action_batch = torch.tensor(batch.action)
    reward_batch = torch.tensor(batch.reward)
    next_state_batch = torch.stack(batch.next_state)

    max_qsa = target_net(next_state_batch).max(dim=-1).values  # [BATCH_SIZE,]
    # Compute the expected Q values
    y_targets = reward_batch + (max_qsa * GAMMA)  # missing terminal state

    q_values = policy_net(state_batch).gather(1, action_batch[:, None])[:, 0]
    loss = criterion(q_values, y_targets)
    print('- loss: ', loss.detach().item())

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def train(env: Environment):
    time.sleep(3)

    action = 2
    state = process_raw_input(env.step(action)).to(device)  # [14, 26, 11]
    counter = 0

    for t in count():
        counter += 1
        # Select and perform an action
        action = select_action(state)[0].item()
        raw_data = env.step(action)
        next_state = process_raw_input(raw_data).to(device)  # [14, 26, 11]
        reward = compute_reward(raw_data)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        if not EVAL_MODE:
            optimize_model()
        # Update the target network, copying all weights and biases in DQN
        if counter % TARGET_UPDATE == 0:
            print('update target network')
            # target_net.load_state_dict(policy_net.state_dict())
            target_net.load_state_dict(policy_net.state_dict())
            # save_model
            torch.save(policy_net.state_dict(), './model/dqn.pth')
