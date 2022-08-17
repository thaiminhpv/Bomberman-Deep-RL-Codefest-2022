import math
import os
import random
import time
from itertools import count
from threading import Thread
from typing import Tuple

import numpy as np
import torch
from torch import optim

from drl.DQN import *
from drl.Environment import Environment
from drl.ReplayMemory import ReplayMemory, Transition
from drl.preprocessing import process_raw_input, compute_reward
from drl.util import seed_everything, plot_loss, log

EVAL_MODE = False
RESUME = False

RANDOM_SEED = 420

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 700
TARGET_UPDATE = 100
TAU = 0.001
PLOT_INTERVAL = 10

if EVAL_MODE:
    RESUME = True
    EPS_START = 0.05

screen_height = 14
screen_width = 26
depth = 3
n_actions = 6

steps_done = 0
episode_durations = []

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model.pth')

# create model dir
if not os.path.exists(os.path.join(BASE_DIR, 'model')):
    os.makedirs(os.path.join(BASE_DIR, 'model'))

seed_everything(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN(screen_height, screen_width, depth, n_actions).to(device)
target_net = DQN(screen_height, screen_width, depth, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

if RESUME:
    if os.path.isfile(MODEL_PATH):
        print('load model from {}'.format(MODEL_PATH))
        policy_net.load_state_dict(torch.load(MODEL_PATH))
        target_net.load_state_dict(torch.load(MODEL_PATH))
    else:
        print('no model')

optimizer = optim.Adam(policy_net.parameters(), lr=0.0001)
memory = ReplayMemory(10000)
criterion = nn.SmoothL1Loss()

mapping = {0: '←', 1: '→', 2: '↑', 3: '↓', 4: 'bomb', 5: 'stop'}


def select_action(state) -> Tuple[torch.Tensor, float]:
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            policy_net.eval()
            temp = policy_net(state[None, ...])
            policy_net.train()
            confident = F.softmax(temp, dim=1).max(1).values.detach().item()
            print(f'{mapping[temp.argmax().item()]} : {confident * 100:.3f}%')
            return temp.argmax().view(1, 1), confident
            # return policy_net(state[None, ...]).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), 0.0


def optimize_model():
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    confident = torch.tensor(batch.info).to(device).mean()
    state_batch = torch.stack(batch.state).to(device)
    action_batch = torch.tensor(batch.action).to(device)
    reward_batch = torch.tensor(batch.reward).to(device)
    next_state_batch = torch.stack(batch.next_state).to(device)

    max_qsa = target_net(next_state_batch).max(dim=-1).values  # [BATCH_SIZE,]
    # Compute the expected Q values
    y_targets = reward_batch + (max_qsa * GAMMA)  # missing terminal state

    q_values = policy_net(state_batch).gather(1, action_batch[:, None])
    loss = criterion(q_values, y_targets[:, None])

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    loss = loss.detach()
    return loss, confident


def recall_model():
    losses = []
    confidents = []
    total_losses = []
    total_confidents = []
    # wait until len(memory) < BATCH_SIZE
    while len(memory) < BATCH_SIZE:
        print('not enough sample to recall, observing...')
        time.sleep(3)
    print('------------------- start training -------------------')
    for t in count():
        loss, confident = optimize_model()

        losses.append(loss)
        confidents.append(confident)

        # Update the target network, copying all weights and biases in DQN
        if t % TARGET_UPDATE == 0:
            print('update target network')
            target_net.load_state_dict(policy_net.state_dict())
            # perform soft update
            # for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            #     target_param.data.copy_(target_param.data * (1.0 - TAU) + policy_param.data * TAU)

        if t % PLOT_INTERVAL == 0:
            print('saving model...')
            # save_model
            # torch.save(policy_net.state_dict(), MODEL_PATH)

            _losses, _confidents = torch.tensor(losses), torch.tensor(confidents)
            # filter out the invalid data
            _losses = _losses[_confidents > 0.0].mean()
            _confidents = _confidents[_confidents > 0.0].mean()

            if torch.isnan(_losses) or torch.isnan(_confidents):
                continue
            _loss = _losses.item()
            _confident = _confidents.item()
            total_losses.append(_loss)
            total_confidents.append(_confident)

            log(_loss, _confident, t)
            plot_loss(total_losses, total_confidents)

            losses.clear()
            confidents.clear()


def train(env: Environment):
    time.sleep(1)

    action = 2
    state = process_raw_input(env.step(action)).to(device)  # [14, 26, 11]

    if not EVAL_MODE:
        Thread(target=recall_model).start()

    for t in count():
        # Select and perform an action
        _, confident = select_action(state)
        action = _[0].item()
        raw_data = env.step(action)
        next_state = process_raw_input(raw_data).to(device)  # [14, 26, 11]
        reward = compute_reward(raw_data)

        # Store the transition in memory
        memory.push(state, action, next_state, reward, confident)

        # Move to the next state
        state = next_state
