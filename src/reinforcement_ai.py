from threading import Condition
from time import sleep

from drl.Environment import Environment
from src.Hero import Hero
from drl.train import train


class ReinforcementAI(Hero):
    def __init__(self, player_id, verbose=True):
        super(ReinforcementAI, self).__init__(player_id, verbose=verbose)
        self.condition = Condition()
        self.env = Environment(cv=self.condition, player_id=self.player_id, move=lambda _: self.move(_))

    def on_join_game(self, data):
        train(self.env)

    def on_drive_player(self, data):
        pass

    def on_ticktack_player(self, data):
        with self.condition:
            self.env.tick(data)
            self.condition.notify()
