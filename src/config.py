# GAME_ID = "583c964a-dc02-497a-9f38-45af3b15e0db"

SERVER_URL = "https://server.jsclub.me/"


# GAME_ID =

def get_game_id():
    import requests

    url = 'http://localhost:5555/'
    r = requests.get(url)
    return r.text
