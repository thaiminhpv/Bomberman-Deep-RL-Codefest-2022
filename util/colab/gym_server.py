from threading import active_count

import flask
import socketio
from flask import request
from flask_cors import CORS, cross_origin


def GYM_SERVER():
    sio = socketio.Server(async_mode='threading', cors_allowed_origins='*')
    app = flask.Flask(__name__)
    app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)

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
                sio.emit('game_id', {'game_id': game_id})
                return "Game id set to {}".format(game_id)
            else:
                # print total number of threads
                print('number of threads: ', active_count())
                return app.config['GAME_ID']

    @sio.on('connect')
    def connect(sid, environ):
        print('connect ', sid)
        sio.emit('game_id', {'game_id': app.config['GAME_ID']})
        print('game id sent to client')

    @sio.on('disconnect')
    def disconnect(sid):
        print('disconnect ', sid)

    @sio.on('connect_error')
    def connect_error(sid):
        print('The connection has an error')

    app.run(threaded=True, port=5001)


if __name__ == '__main__':
    GYM_SERVER()
