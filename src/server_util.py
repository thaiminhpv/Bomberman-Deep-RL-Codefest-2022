from pprint import pprint
from .config import *


def server_util(sio, player_id):
    @sio.event
    def connect():
        print('connection established')
        sio.emit('join game', {'game_id': GAME_ID, 'player_id': player_id})
        print(f'{player_id} connected to game {GAME_ID}')

    @sio.event
    def disconnect():
        print('disconnected from server')

    @sio.event
    def connect_error():
        print('The connection has an error')

    @sio.on('join game')
    def on_join_game(data):
        print(f'{player_id} join game successfully')
        pprint(data)
