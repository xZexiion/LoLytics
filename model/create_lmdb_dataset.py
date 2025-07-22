import numpy as np
import json

with open('champion_to_index.json') as f:
    champ_to_idx = json.load(f)
dragon_names = ['WATER_DRAGON', 'AIR_DRAGON', 'CHEMTECH_DRAGON', 'FIRE_DRAGON', 'HEXTECH_DRAGON', 'EARTH_DRAGON']

def convert_json_sample_to_numpy(sample):
    l = []

    for team in sample['teams']:
        for player in team['players']:
            for k in player:
                if k == 'champion':
                    l.append(champ_to_idx[player['champion']])
                elif k == 'deathTimer':
                    l.append(round(player['deathTimer']))
                else:
                    l.append(player[k])
        for i in range(4):
            one_hot = [0, 0, 0, 0, 0, 0]
            if i < len(team['drakes']):
                name = team['drakes'][i]
                if name == 'ELDER_DRAGON':
                    continue
                one_hot[dragon_names.index(name)] = 1
            l += one_hot
        l.append(team['rifts'])
        l.append(team['grubs'])
        l += team['towers']
        l += team['inhibs']
    l.append(sample['time'])
    l.append(sample['win'])

    return np.array(l, dtype='int32')

with open('../data/match_data/PLATINUM/game_230/5.json') as f:
    x = json.load(f)
    y = convert_json_sample_to_numpy(x)
    print(y.dtype)
    print(y.shape)
    print(y)
    print(len(y))