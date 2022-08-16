from pprint import pprint
from .config import SERVER_URL, get_game_id


def server_util(sio, player_id, verbose=True):
    log = print if verbose else lambda *args, **kwargs: None

    @sio.event
    def connect():
        log('connection established')
        print('3. server responded connected successfully - ready to emit join game')
        sio.emit('join game', {'game_id': get_game_id(), 'player_id': player_id})
        print('4. done emitted join game')
        log(f'{player_id} connected to game {get_game_id()}')

    @sio.event
    def disconnect():
        log('disconnected from server')

    @sio.event
    def connect_error():
        log('The connection has an error')

    return log
