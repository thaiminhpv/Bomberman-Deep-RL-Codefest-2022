from src.reinforcement_ai import ReinforcementAI
from src.random_bot import RandomBot
import time

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

    player1 = RandomBot('player1-xxx', game_id, verbose=False)
    player2 = ReinforcementAI('player2-xxx', game_id, verbose=False)
    player1.run()
    player2.run()
