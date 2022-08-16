from abc import ABC, abstractmethod
import socketio
from pprint import pprint
import random
from src.config import SERVER_URL
from src.server_util import server_util


class Hero(ABC):
    MAPPING = {0: '1', 1: '2', 2: '3', 3: '4', 4: 'b', 5: 'x'}

    def __init__(self, player_id, verbose=True):
        super(Hero, self).__init__()
        self.sio = socketio.Client()
        self.player_id = player_id
        self.verbose = verbose
        self.running = False
        self.log = server_util(self.sio, self.player_id, verbose=self.verbose)
        self.log(f'{self.player_id} started')

        @self.sio.on('drive player')
        def on_drive_player(data):
            # self.log(f'{self.player_id} drive player successfully')
            self.on_drive_player(data)

        @self.sio.on('ticktack player')
        def on_ticktack_player(data):
            # self.log(f'{self.player_id} received ticktack player')
            self.on_ticktack_player(data)

        @self.sio.on('join game')
        def on_join_game(data):
            self.log(f'{player_id} join game successfully')
            print('5. server call back responded joined game successfully - ready to emit drive player')
            pprint(data)
            self.running = True
            print('\trunning = ', self.running)
            print('6. running variable set')
            self.on_join_game(data)

    def move(self, step):
        if isinstance(step, str):
            step = step
        elif isinstance(step, int):
            step = Hero.MAPPING[step]
        else:
            raise TypeError('step must be str or int')
        print(f'{self.player_id} move: {step}')
        print('\trunning = ', self.running)
        print('self.sio.connected = ', self.sio.connected)
        print('self: ', self)
        assert self.running, 'game is not running'
        self.log(f'Player = {self.player_id} - Dir = {step}')
        self.sio.emit('drive player', {"direction": step})

    @abstractmethod
    def on_drive_player(self, data):
        pass

    @abstractmethod
    def on_ticktack_player(self, data):
        pass

    @abstractmethod
    def on_join_game(self, data):
        raise NotImplementedError('on_join_game must be implemented')
        pass

    def run(self):
        print(f'1. {self.player_id} ready connect to server')
        self.sio.connect(SERVER_URL)
        print(f'2. {self.player_id} initiated connection to server')

    def __del__(self):
        self.sio.disconnect()
        self.log(f'{self.player_id} stopped')
        self.running = False
