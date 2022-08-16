import socketio
from threading import Thread
import os
import time
import sys

# append src, util, drl to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(parent_dir + '/src')
sys.path.append(parent_dir + '/util')
sys.path.append(parent_dir + '/drl')
print(sys.path)

from src.run_agents import run_agents
SERVER = 'https://ai.jsclub.me/'


def GYM_CLIENT():
    sio = socketio.Client()

    @sio.on('connect')
    def on_connect():
        print('connected to server')

    @sio.on('disconnect')
    def on_disconnect():
        print('disconnected from server')

    @sio.on('game_id')
    def on_game_id(data):
        game_id = data['game_id']
        if game_id:
            print('game_id:', game_id)
            Thread(target=run_agents, args=(game_id,)).start()
        else:
            print('game_id is empty')

    sio.connect(SERVER)


if __name__ == '__main__':
    GYM_CLIENT()
