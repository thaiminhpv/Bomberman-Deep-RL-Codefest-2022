started = False

from threading import Condition
from threading import Thread

import socketio

from drl.Environment import Environment
from drl.train import train
from src.config import SERVER_URL
from src.server_util import server_util

mapping = {0: '1', 1: '2', 2: '3', 3: '4', 4: 'b', 5: 'x'}


def reinforcement_ai(player_id: str):
    cv = Condition()
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

    Thread(target=train).start()
    sio.connect(SERVER_URL)
