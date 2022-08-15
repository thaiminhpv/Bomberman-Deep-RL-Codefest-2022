# Singleton Flask app that set variable GAME_ID  listening on 5555
import flask
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin


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
            if game_id is not None:
                print('game_id: ', game_id)
                app.config['GAME_ID'] = game_id
                return "Game id set to {}".format(game_id)
            else:
                return app.config['GAME_ID']

    app.run(host='localhost', port=5555)


if __name__ == '__main__':
    GYM()
