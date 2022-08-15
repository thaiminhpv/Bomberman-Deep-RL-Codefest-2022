import math
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

RANDOM_SEED = 420

BATCH_SIZE = 32
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 100

screen_height = 14
screen_width = 26
depth = 11
n_actions = 6

steps_done = 0
episode_durations = []

seed_everything(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN(screen_height, screen_width, depth, n_actions).to(device)
target_net = DQN(screen_height, screen_width, depth, n_actions).to(device)
target_net.target.load_state_dict(policy_net.online.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.0001)
memory = ReplayMemory(10000)


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state[None, ...], model='online').max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                  dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])
    state_batch = torch.stack(batch.state)
    action_batch = torch.tensor(batch.action)
    reward_batch = torch.tensor(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch, model='online').gather(1, action_batch[:, None])[:, 0]


    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    # next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values = target_net(non_final_next_states, model='target').max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)
    print('loss: ', loss.detach().item())

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.online.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def train():
    time.sleep(3)
    env = Environment()

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
        optimize_model()
        # Update the target network, copying all weights and biases in DQN
        if counter % TARGET_UPDATE == 0:
            print('update target network')
            # target_net.load_state_dict(policy_net.state_dict())
            target_net.target.load_state_dict(policy_net.online.state_dict())
