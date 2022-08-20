# Singleton Flask app that set variable GAME_ID  listening on 5555
from threading import Thread, active_count
import os
import sys

import flask
from flask import request
from flask_cors import CORS, cross_origin

# append src, util, drl to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(parent_dir + '/src')
sys.path.append(parent_dir + '/util')
sys.path.append(parent_dir + '/drl')
print(sys.path)

from src.run_agents import run_agents


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
                # Thread(target=run_agents, args=(game_id,)).start()
                return "Game id set to {}".format(game_id)
            else:
                # print total number of threads
                print('number of threads: ', active_count())
                return app.config['GAME_ID']

    app.run(threaded=True, host='0.0.0.0', port=5555)


if __name__ == '__main__':
    GYM()
