import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from drl.Environment import Environment


def process_raw_input(data) -> torch.Tensor:
    mapp = data['map_info']['map']
    mapp = torch.tensor(mapp)
    mapp[mapp == 6] = 3
    mapp[mapp == 7] = 4

    mapp = F.one_hot(mapp)
    # %%
    spoils = data['map_info']['spoils']
    map_spoils = torch.zeros(data['map_info']['size']['rows'], data['map_info']['size']['cols'])
    for spoil in spoils:
        map_spoils[spoil['row'], spoil['col']] = int(spoil['spoil_type'])
    map_spoils = map_spoils.long()
    map_spoils[map_spoils == 3] = 0
    map_spoils[map_spoils == 4] = 1
    map_spoils[map_spoils == 5] = 2
    one_hot_map_spoils = F.one_hot(map_spoils, num_classes=3)

    # %%
    MAX_REMAINING_TIME = 2000
    bombs = data['map_info']['bombs']
    map_bombs_power = torch.zeros(data['map_info']['size']['rows'], data['map_info']['size']['cols'])
    for i in range(len(data['map_info']['players'])):
        power = data['map_info']['players'][i]['power']
        player_id = data['map_info']['players'][i]['id']
        for bomb in bombs:
            if bomb['playerId'] == player_id:
                remainTime = (MAX_REMAINING_TIME - int(bomb['remainTime'])) / 2000
                # map_bombs_power[bomb['row'], bomb['col']] = remainTime
                # print(power)
                for p in range(power + 1):
                    _row = min(bomb['row'] + p, data['map_info']['size']['rows'] - 1)
                    _col = min(bomb['col'] + p, data['map_info']['size']['cols'] - 1)
                    map_bombs_power[_row, bomb['col']] = remainTime
                    map_bombs_power[bomb['row'], _col] = remainTime
                    _row = max(0, bomb['row'] - p)
                    _col = max(0, bomb['col'] - p)
                    map_bombs_power[_row, bomb['col']] = remainTime
                    map_bombs_power[bomb['row'], _col] = remainTime

    player_id = Environment().player_id  # data['player_id']
    map_enemy = torch.zeros(data['map_info']['size']['rows'], data['map_info']['size']['cols'])
    for i in range(len(data['map_info']['players'])):
        if data['map_info']['players'][i]['id'] != player_id:
            player_id = data['map_info']['players'][i]['id']
            map_enemy[data['map_info']['players'][i]['currentPosition']['row'], data['map_info']['players'][i]['currentPosition']['col']] = 1
            # add 1 to left, right, up, down
            map_enemy[data['map_info']['players'][i]['currentPosition']['row'], data['map_info']['players'][i]['currentPosition']['col'] - 1] = 1
            map_enemy[data['map_info']['players'][i]['currentPosition']['row'], data['map_info']['players'][i]['currentPosition']['col'] + 1] = 1
            map_enemy[data['map_info']['players'][i]['currentPosition']['row'] - 1, data['map_info']['players'][i]['currentPosition']['col']] = 1
            map_enemy[data['map_info']['players'][i]['currentPosition']['row'] + 1, data['map_info']['players'][i]['currentPosition']['col']] = 1
    map_current_player = torch.zeros(data['map_info']['size']['rows'], data['map_info']['size']['cols'])
    for i in range(len(data['map_info']['players'])):
        if data['map_info']['players'][i]['id'] == player_id:
            player_id = data['map_info']['players'][i]['id']
            map_current_player[data['map_info']['players'][i]['currentPosition']['row'], data['map_info']['players'][i]['currentPosition']['col']] = 1
            # add 1 to left, right, up, down
            map_current_player[data['map_info']['players'][i]['currentPosition']['row'], data['map_info']['players'][i]['currentPosition']['col'] - 1] = 1
            map_current_player[data['map_info']['players'][i]['currentPosition']['row'], data['map_info']['players'][i]['currentPosition']['col'] + 1] = 1
            map_current_player[data['map_info']['players'][i]['currentPosition']['row'] - 1, data['map_info']['players'][i]['currentPosition']['col']] = 1
            map_current_player[data['map_info']['players'][i]['currentPosition']['row'] + 1, data['map_info']['players'][i]['currentPosition']['col']] = 1

    map_human = torch.zeros(data['map_info']['size']['rows'], data['map_info']['size']['cols'])
    for human in data['map_info']['human']:
        position = human['position']
        if human['infected']:
            map_human[position['row'], position['col']] = 1
            direction = human.get('direction', None)
            if direction == 1:  # left
                map_human[position['row'], position['col'] - 1] = 1
            elif direction == 2:  # right
                map_human[position['row'], position['col'] + 1] = 1
            elif direction == 3:  # up
                map_human[position['row'] - 1, position['col']] = 1
            elif direction == 4:  # down
                map_human[position['row'] + 1, position['col']] = 1
    # %%

    map_virus = torch.zeros(data['map_info']['size']['rows'], data['map_info']['size']['cols'])
    for virus in data['map_info']['viruses']:
        position = virus['position']
        direction = virus['direction']
        map_virus[position['row'], position['col']] = 1
        if direction == 1:  # left
            map_virus[position['row'], position['col'] - 1] = 1
        elif direction == 2:  # right
            map_virus[position['row'], position['col'] + 1] = 1
        elif direction == 3:  # up
            map_virus[position['row'] - 1, position['col']] = 1
        elif direction == 4:  # down
            map_virus[position['row'] + 1, position['col']] = 1

    map_all = torch.cat((mapp, one_hot_map_spoils, map_bombs_power[..., None], map_enemy[..., None], map_current_player[..., None], map_human[..., None], map_virus[..., None]), dim=2)
    # [14, 26, 13]
    return map_all.float()


previous = {}
previous['score'], previous['lives'], previous['pill'], previous['power'], previous['quarantine'], previous[
    'humanCured'], previous['humanSaved'] = 0, 1000, 0, 1, 0, 0, 0


def compute_reward(data):
    info = None
    player_id = Environment().player_id
    for i in range(len(data['map_info']['players'])):
        if data['map_info']['players'][i]['id'] == player_id:
            player_id = data['map_info']['players'][i]['id']
            info = data['map_info']['players'][i]
            break
    else:
        raise Exception('player_id not found')

    # compute difference between previous and current
    score_diff = info['score'] - previous['score']
    lives_diff = info['lives'] - previous['lives']
    pill_diff = info['pill'] - previous['pill']
    power_diff = info['power'] - previous['power']
    quarantine_diff = info['quarantine'] - previous['quarantine']
    humanCured_diff = info['humanCured'] - previous['humanCured']
    humanSaved_diff = info['humanSaved'] - previous['humanSaved']

    # compute reward
    reward = score_diff * 3 + lives_diff * 100 + pill_diff + power_diff * 15 + quarantine_diff * -40 + humanCured_diff * 3 + humanSaved_diff * 3 - 0.5

    # print different if != 0
    if lives_diff >= -1:
        for key, value in previous.items():
            if value != info[key]:
                print(
                    f"{key}: {value} -> {info[key]} : {'+' if info[key] - value > 0 else ''}{info[key] - value} : REWARD: {reward}")

    previous['score'], previous['lives'], previous['pill'], previous['power'], previous['quarantine'], previous[
        'humanCured'], previous['humanSaved'] = info['score'], info['lives'], info['pill'], info['power'], info[
        'quarantine'], info['humanCured'], info['humanSaved']

    if quarantine_diff == 1:
        print(f"Got to quarantine")
        time.sleep(14)
        print(f"Got out of quarantine")

    if lives_diff < -1:
        return 1
    return reward

