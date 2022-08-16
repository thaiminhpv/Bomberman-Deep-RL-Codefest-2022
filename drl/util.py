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


def log(losses, confidents, time_steps):
    logger = SummaryWriter('runs/drl-bot-Codefest')

    print(f'\t\t\t\t\t| loss: {losses:.4f} - conf: {confidents:.4f}')

    # add to tensorboard
    logger.add_scalar('loss', losses, time_steps)
    logger.add_scalar('confident', confidents, time_steps)

    logger.close()


def plot_loss(losses, confidents):
    if len(losses) < 2:
        return
    # _losses = losses.unfold(0, 6, 1).mean(1).view(-1)
    # _confidents = confidents.unfold(0, 6, 1).mean(1).view(-1)
    # plot loss and confidents on the same figure with different colors and their own scale
    fig, ax = plt.subplots()
    l1, = ax.plot(losses, color='red', label='loss')
    ax.set_xlabel('time steps')
    ax.set_ylabel('loss')
    ax2 = ax.twinx()
    l2, = ax2.plot(confidents, color='blue', label='confident')
    ax2.set_ylabel('confident')
    plt.legend(handles=[l1, l2], loc='upper right')
    plt.show()
