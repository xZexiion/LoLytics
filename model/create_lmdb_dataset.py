import numpy as np
import json
import lmdb
import pickle
from tqdm import tqdm
import os
import random

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
        l.append(team['atakhan'])
        l.append(team['grubs'])
        l += team['towers']
        l += team['inhibs']
    l.append(sample['time'])
    l.append(sample['win'])

    return np.array(l, dtype='int32')

def get_file_paths(data_dir):
    paths = []

    for dirname, _, file_names in tqdm(os.walk('../match_data')):
        for file_name in file_names:
            if file_name == '.DS_Store':
                continue
            path = os.path.join(dirname, file_name)
            paths.append(path)

    return paths
    
def save_lmdb(paths, lmdb_path):
    map_size = 1 << 40
    env = lmdb.open(lmdb_path, map_size=map_size)

    idx = 0
    with env.begin(write=True) as txn:
        for path in paths:
            with open(path) as f:
                obj = json.load(f)
                data = convert_json_sample_to_numpy(obj)
            key = str(idx).encode()
            value = pickle.dumps(data)
            txn.put(key, value)
            idx += 1

def main():
    file_paths = get_paths()
    random.shuffle(file_paths)

    split = 0.8

    num_train_samples = int(len(file_paths) * split)
    train_paths = file_paths[:num_train_samples]
    test_paths = file_paths[num_train_samples:]

    save_lmdb(train_paths, 'train.lmdb')
    save_lmdb(test_paths, 'test.lmdb')
    

main()
