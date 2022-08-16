import socketio
import asyncio
from pprint import pprint
import random
from src.config import SERVER_URL
from src.server_util import server_util
from src.Hero import Hero


class RandomBot(Hero):

    def __init__(self, player_id, game_id, verbose=True):
        super(RandomBot, self).__init__(player_id, game_id, verbose=verbose)

    def on_join_game(self, data):
        pass

    def on_drive_player(self, data):
        pass

    def on_ticktack_player(self, data):
        self.move(step=''.join(random.choice(['1', '2', '3', '4', 'b', 'x']) for _ in range(4)))


if __name__ == '__main__':
    RandomBot('player2-xxx').run()
