import requests

url = 'http://localhost:5555/'
r = requests.get(url)
# GAME_ID = "583c964a-dc02-497a-9f38-45af3b15e0db"

SERVER_URL = "https://localhost.jsclub.me/"
GAME_ID = r.text
PLAYER_ID = "player1-xxx"
