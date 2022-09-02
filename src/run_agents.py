import os
import time
import sys
# from src.do_nothing_bot import NothingBot
# from src.reinforcement_ai import ReinforcementAI

# append src, util, drl to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(parent_dir + '/src')
sys.path.append(parent_dir + '/util')
sys.path.append(parent_dir + '/drl')
print(sys.path)

from src.random_bot import RandomBot
from src.do_nothing_bot import NothingBot

player1 = None
player2 = None


def run_agents(game_id: str):
    """
    If agent is already running, kill it and start a new one
    :return:
    """
    global player1, player2
    time.sleep(0.1)

    if player1 is not None:
        print('player1 is running, kill it')
        if player1.running:
            print('killing player1')
            player1.__del__()
            del player1
    if player2 is not None:
        print('player2 is running, kill it')
        if player2.running:
            print('killing player2')
            player2.__del__()
            del player2

    time.sleep(1)

    # player1 = RandomBot('player1-xxx', game_id, verbose=False)
    # player1 = RandomBot('player1-xxx', game_id, verbose=False)
    # player2 = RandomBot('player1-xxx', game_id, verbose=False)
    player1 = RandomBot('9e1b0a03-2f8c-4e00-86f8-fa81dd41ad04', game_id, verbose=False)
    player2 = RandomBot('bc7b124e-ac31-499b-86b0-fc50932f6ba8', game_id, verbose=False)
    # player2 = NothingBot('01351af5-4873-4044-aeee-97a1553e408c', game_id, verbose=False)
    player1.run()
    player2.run()


if __name__ == '__main__':
    # run_agents('969bb2a8-c32d-4ee5-a0b6-77d9b2d6c04a')
    run_agents('c283a5e5-3706-4b4c-a564-344485201272')

