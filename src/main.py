import sys

import socketio
from pprint import pprint, pformat
import random
from src.config import *
from src.server_util import server_util

PLAYER_ID = "player1-xxx"

store_data = None

sio = socketio.Client()

server_util(sio, PLAYER_ID)


def move(step):
    # take a random character from the list of characters
    step = random.choice(['1', '2', '3', '4', 'b', 'x'])
    print(f'Player = {PLAYER_ID} - Dir = {step}')
    sio.emit('drive player', {"direction": step})


@sio.on('drive player')
def on_drive_player(data):
    print(f'{PLAYER_ID} drive player successfully')
    pprint(data)


@sio.on('ticktack player')
def on_ticktack_player(data):
    print(f'{PLAYER_ID} received ticktack player')
    data = pformat(data, indent=4)
    # dump data to file
    with open('data.json', 'w') as f:
        f.write(str(data))
    sys.exit(0)
    pprint(data)
    move('1234b')


sio.connect(SERVER_URL)
