import socketio
import asyncio
from pprint import pprint
import random
from src.config import SERVER_URL
from src.server_util import server_util
from src.Hero import Hero

SEED = 420

random.seed(SEED)


class NothingBot(Hero):

    def __init__(self, player_id, game_id, verbose=True):
        super(NothingBot, self).__init__(player_id, game_id, verbose=verbose)

    def on_join_game(self, data):
        pass

    def on_drive_player(self, data):
        pass

    def on_ticktack_player(self, data):
        pass


if __name__ == '__main__':
    NothingBot('player2-xxx', '2f561e29-0628-4d11-8bf7-cda45a523203').run()
