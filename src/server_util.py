from pprint import pprint
from .config import *


def server_util(sio, player_id, verbose=True):
    log = print if verbose else lambda *args, **kwargs: None

    @sio.event
    def connect():
        log('connection established')
        sio.emit('join game', {'game_id': GAME_ID, 'player_id': player_id})
        log(f'{player_id} connected to game {GAME_ID}')

    @sio.event
    def disconnect():
        log('disconnected from server')

    @sio.event
    def connect_error():
        log('The connection has an error')

    @sio.on('join game')
    def on_join_game(data):
        log(f'{player_id} join game successfully')
        pprint(data)

    return log
