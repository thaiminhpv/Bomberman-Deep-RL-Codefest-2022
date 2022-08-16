import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def seed_everything(seed):
    seed = int(seed)
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def plot_durations(episode_durations):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    # if is_ipython:
    #     display.clear_output(wait=True)
    #     display.display(plt.gcf())


def plot_loss(losses, confidents, time_steps: int):
    logger = SummaryWriter('runs/drl-bot-Codefest')
    INTERVAL = 40
    print('plot loss')

    _losses = np.array(losses)
    mean_losses = np.stack(np.split(_losses[_losses.shape[0] % INTERVAL:], INTERVAL)).mean(axis=0)
    _confidents = np.array(confidents)
    mean_confidents = np.stack(np.split(_confidents[_confidents.shape[0] % INTERVAL:], INTERVAL)).mean(axis=0)

    # filter out 0 values of confidents
    mean_confidents = mean_confidents[mean_confidents > 0]
    mean_losses = mean_losses[mean_confidents > 0]

    # add to tensorboard
    logger.add_scalar('loss', mean_losses.mean(), time_steps)
    logger.add_scalar('confident', mean_confidents.mean(), time_steps)

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
