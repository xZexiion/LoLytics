import numpy as np
import json
import lmdb
import pickle
from tqdm import tqdm
import os

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
                one_hot[dragon_names.index(name)] = 1
            l += one_hot
        l.append(team['rifts'])
        l.append(team['grubs'])
        l += team['towers']
        l += team['inhibs']
    l.append(sample['time'])
    l.append(sample['win'])

    return np.array(l, dtype='int32')

def main():
    lmdb_path = "dataset.lmdb"
    map_size = 1 << 4

    env = lmdb.open(lmdb_path, map_size=map_size)

    with env.begin(write=True) as txn:
        for i, (dirname, _, file_names) in tqdm(enumerate(os.walk('../data/match_data'))):
            for file_name in file_names:
                path = os.path.join(dirname, file_name)
                with open(path) as f:
                    obj = json.load(f)
                    data = convert_json_sample_to_numpy(obj)
                key = str(i).encode()
                value = pickle.dumps(data)
                txn.put(key, value)

main()
