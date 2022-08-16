import os
# import asyncio
from typing import Tuple

import matplotlib.pyplot as plt

from drl.ReplayMemory import *
from drl.preprocessing import *
from drl.util import *

EVAL_MODE = False
resume = False

RANDOM_SEED = 420

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 600
TARGET_UPDATE = 300

if EVAL_MODE:
    resume = True
    EPS_START = 0.05

screen_height = 14
screen_width = 26
depth = 5
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
    if os.path.isfile('../model/dqn.pth'):
        print('load model')
        policy_net.load_state_dict(torch.load('../model/dqn.pth'))
        target_net.load_state_dict(torch.load('../model/dqn.pth'))
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
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward
            temp = policy_net(state[None, ...])
            confident = F.softmax(temp, dim=1).max(1).values.detach().item()
            print(f'{mapping[temp.argmax().item()]} : {confident * 100:.3f}%')
            return temp.argmax().view(1, 1), confident
            # return policy_net(state[None, ...]).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), 0.0


def optimize_model():
    policy_net.train()
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.stack(batch.state)
    action_batch = torch.tensor(batch.action)
    reward_batch = torch.tensor(batch.reward)
    next_state_batch = torch.stack(batch.next_state)

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

    return loss.item()


def plot_loss(losses, confidents, time_steps: int):
    INTERVAL = 40
    print('plot loss')

    _losses = np.array(losses)
    mean_losses = np.stack(np.split(_losses[_losses.shape[0] % INTERVAL:], INTERVAL)).mean(axis=0)
    _confidents = np.array(confidents)
    mean_confidents = np.stack(np.split(_confidents[_confidents.shape[0] % INTERVAL:], INTERVAL)).mean(axis=0)

    # filter out 0 values of confidents
    mean_confidents = mean_confidents[mean_confidents > 0]
    mean_losses = mean_losses[mean_confidents > 0]

    # plot loss and confidents on the same figure with different colors and their own scale
    fig, ax = plt.subplots()
    ax.plot(mean_losses, color='red', label='loss')
    ax.set_xlabel('time steps')
    ax.set_ylabel('loss')
    ax2 = ax.twinx()
    ax2.plot(mean_confidents, color='blue', label='confident')
    ax2.set_ylabel('confident')
    plt.legend()
    plt.show()


def train(env: Environment):
    time.sleep(3)

    action = 2
    state = process_raw_input(env.step(action)).to(device)  # [14, 26, 11]
    counter = 0
    losses = []
    confidents = []

    for t in count():
        counter += 1
        # Select and perform an action
        _, confident = select_action(state)
        action = _[0].item()
        raw_data = env.step(action)
        next_state = process_raw_input(raw_data).to(device)  # [14, 26, 11]
        reward = compute_reward(raw_data)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        if not EVAL_MODE:
            if len(memory) < BATCH_SIZE:
                print('--- not enough sample to recall, observing... ---')
            else:
                loss = optimize_model()

                losses.append(loss)
                confidents.append(confident)

            # Update the target network, copying all weights and biases in DQN
            if counter % TARGET_UPDATE == 0:
                print('update target network')
                # target_net.load_state_dict(policy_net.state_dict())
                # perform soft update
                TAU = 0.001
                for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - TAU) + policy_param.data * TAU)

                # save_model
                torch.save(policy_net.state_dict(), '../model/dqn.pth')

                plot_loss(losses, confidents, t)
