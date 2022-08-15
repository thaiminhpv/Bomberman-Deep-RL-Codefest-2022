from threading import Condition
from drl.Environment import *
from src.reinforcement_ai import reinforcement_ai
from src.config import *


if __name__ == '__main__':
    reinforcement_ai(Condition(), 'player1-xxx')
