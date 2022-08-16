# Singleton Flask app that set variable GAME_ID  listening on 5555
import time
from threading import Thread, active_count
import flask
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
from src.reinforcement_ai import ReinforcementAI
from src.random_bot import RandomBot

player1 = None
player2 = None

def run_agents():
    """
    If agent is already running, kill it and start a new one
    :return:
    """
    global player1, player2
    time.sleep(0.1)

    if player1:
        print('player1 is running, kill it')
        if player1.running:
            print('killing player1')
            player1.__del__()
            del player1
    if player2:
        print('player2 is running, kill it')
        if player2.running:
            print('killing player2')
            player2.__del__()
            del player2

    time.sleep(1)

    player1 = RandomBot('player1-xxx', verbose=False)
    player2 = ReinforcementAI('player2-xxx', verbose=False)
    player1.run()
    player2.run()


def GYM():
    app = flask.Flask(__name__)
    cors = CORS(app)
    app.config['CORS_HEADERS'] = 'Content-Type'
    app.config['GAME_ID'] = ""

    @app.route('/', methods=['GET'])
    @cross_origin()
    def set_game_id():
        if request.method == 'GET':
            game_id = request.args.get('game_id')
            if game_id is not None and game_id != "" and game_id.strip() != "":
                print('game_id: ', game_id)
                app.config['GAME_ID'] = game_id
                Thread(target=run_agents).start()
                return "Game id set to {}".format(game_id)
            else:
                # print total number of threads
                print('number of threads: ', active_count())
                return app.config['GAME_ID']

    app.run(threaded=True, host='localhost', port=5555)


if __name__ == '__main__':
    GYM()
