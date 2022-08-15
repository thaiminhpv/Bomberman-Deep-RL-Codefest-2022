import socketio
from pprint import pprint
import random
from src.config import SERVER_URL
from src.server_util import server_util


def RandomBot(player_id: str):
    sio = socketio.Client()
    server_util(sio, player_id, verbose=False)
    sio.connect(SERVER_URL)

    def move(step):
        # take a random character from the list of characters
        step = random.choice(['1', '2', '3', '4', 'b', 'x'])
        # print(f'Player = {player_id} - Dir = {step}')
        sio.emit('drive player', {"direction": step})

    @sio.on('drive player')
    def on_drive_player(data):
        ...
        # print(f'{player_id} drive player successfully')
        # pprint(data)

    @sio.on('ticktack player')
    def on_ticktack_player(data):
        # print(f'{player_id} received ticktack player')
        # pprint(data)
        move('1234b')


if __name__ == '__main__':
    RandomBot('player2-xxx')
