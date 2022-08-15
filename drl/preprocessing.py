import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.nn.functional as F


def process_raw_input(data):
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
    bombs = data['map_info']['bombs']
    map_bombs_power = torch.zeros(data['map_info']['size']['rows'], data['map_info']['size']['cols'])
    for i in range(len(data['map_info']['players'])):
        power = data['map_info']['players'][i]['power']
        player_id = data['map_info']['players'][i]['id']
        for bomb in bombs:
            if bomb['playerId'] == player_id:
                remainTime = int(bomb['remainTime'])
                # map_bombs_power[bomb['row'], bomb['col']] = remainTime
                print(power)
                for p in range(power + 3):
                    map_bombs_power[bomb['row'] + p, bomb['col']] = remainTime
                    map_bombs_power[bomb['row'], bomb['col'] + p] = remainTime
                    _row = max(0, bomb['row'] - p)
                    _col = max(0, bomb['col'] - p)
                    map_bombs_power[_row, bomb['col']] = remainTime
                    map_bombs_power[bomb['row'], _col] = remainTime

    player_id = data['player_id']
    map_enemy = torch.zeros(data['map_info']['size']['rows'], data['map_info']['size']['cols'])
    for i in range(len(data['map_info']['players'])):
        if data['map_info']['players'][i]['id'] != player_id:
            player_id = data['map_info']['players'][i]['id']
            map_enemy[data['map_info']['players'][i]['currentPosition']['row'],
                      data['map_info']['players'][i]['currentPosition']['col']] = 1
    map_current_player = torch.zeros(data['map_info']['size']['rows'], data['map_info']['size']['cols'])
    for i in range(len(data['map_info']['players'])):
        if data['map_info']['players'][i]['id'] == player_id:
            player_id = data['map_info']['players'][i]['id']
            map_current_player[data['map_info']['players'][i]['currentPosition']['row'],
                               data['map_info']['players'][i]['currentPosition']['col']] = 1
    # concat everything
    print('mapp.shape: ', mapp.shape)
    print('one_hot_map_spoils.shape: ', one_hot_map_spoils.shape)
    print('map_bombs_power.shape: ', map_bombs_power.shape)
    print('map_enemy.shape: ', map_enemy.shape)
    print('map_current_player.shape: ', map_current_player.shape)
    # %%
    map_all = torch.cat(
        (mapp, one_hot_map_spoils, map_bombs_power[..., None], map_enemy[..., None], map_current_player[..., None]),
        dim=2)

    return map_all
