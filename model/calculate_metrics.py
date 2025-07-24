import torch
from dataset import Dataset
import numpy as np

dataset = Dataset('dataset.lmdb', True, 1.0)

kills = []
deaths = []
assists = []
gold = []
creepscore = []

data_iter = iter(dataset)

print(len(dataset))

for i in range(len(dataset)):
    sample = next(data_iter)
    for team_offset in [0, 12*5+24+3+11+3]:
        for j in range(5):
            player_offset = j * 12
            offset = team_offset + player_offset

            kills.append(sample[offset + 1].item())
            deaths.append(sample[offset + 2].item())
            assists.append(sample[offset + 3].item())
            gold.append(sample[offset + 7].item())
            creepscore.append(sample[offset + 9].item())

kills = np.array(kills)
deaths = np.array(deaths)
assists = np.array(assists)
gold = np.array(gold)
creepscore = np.array(creepscore)

print(f'Kills | std: {kills.std()} mean: {kills.mean()}')
print(f'Deaths | std: {deaths.std()} mean: {deaths.mean()}')
print(f'Assists | std: {assists.std()} mean: {assists.mean()}')
print(f'Gold | std: {gold.std()} mean: {gold.mean()}')
print(f'CS | std: {creepscore.std()} mean: {creepscore.mean()}')
