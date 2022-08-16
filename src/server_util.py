def server_util(sio, player_id, game_id, verbose=True):
    log = print if verbose else lambda *args, **kwargs: None

    @sio.event
    def connect():
        log('connection established')
        sio.emit('join game', {'game_id': game_id, 'player_id': player_id})
        log(f'{player_id} connected to game {game_id}')

    @sio.event
    def disconnect():
        log('disconnected from server')

    @sio.event
    def connect_error():
        log('The connection has an error')

    return log
