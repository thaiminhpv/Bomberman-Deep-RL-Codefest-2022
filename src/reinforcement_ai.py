started = False

import sys
from threading import Thread, Condition
from drl.train import train

import socketio
from pprint import pprint, pformat
import random
from src.config import *
from src.server_util import server_util
from drl.Environment import Environment

mapping = {0: '1', 1: '2', 2: '3', 3: '4', 4: 'b', 5: 'x'}


def reinforcement_ai(cv: Condition, player_id: str):
    sio = socketio.Client()

    log = server_util(sio, player_id, verbose=False)

    def move(step: int):
        step = mapping[step]
        log(f'Player = {player_id} - Dir = {step}')
        sio.emit('drive player', {"direction": step})

    env = Environment(cv=cv, move=move, player_id=player_id)

    @sio.on('drive player')
    def on_drive_player(data):
        log(f'{player_id} drive player successfully')
        # pprint(data)

    @sio.on('ticktack player')
    def on_ticktack_player(data):
        global started
        started = True
        log(f'{player_id} received ticktack player')
        with cv:
            env.tick(data)
            cv.notify()
        # movement = train_one_episode(data, move_function=move)
        # move(movement)

        # if len(data['map_info']['bombs']) > 0:
        #     pprint(data['map_info']['bombs'][0]['remainTime'])

    Thread(target=train).start()
    sio.connect(SERVER_URL)
