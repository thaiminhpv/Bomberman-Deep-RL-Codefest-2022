from threading import Thread, Condition
from drl.Environment import *
from src.reinforcement_ai import reinforcement_ai


if __name__ == '__main__':
    cv = Condition()
    reinforcement_ai(cv)
