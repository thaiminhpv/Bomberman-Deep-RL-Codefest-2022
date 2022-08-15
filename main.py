from threading import Thread

if __name__ == '__main__':
    from src.reinforcement_ai import reinforcement_ai

    Thread(target=reinforcement_ai, args=('player1-xxx',)).start()

    from src.random_bot import RandomBot

    Thread(target=RandomBot, args=('player2-xxx',)).start()
