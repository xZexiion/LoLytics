from dataset import Dataset
from dnn import DNN
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using {device}')

net = DNN()
net = net.to(device)

dataset = Dataset('dataset.lmdb', True, 0.8)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

samples = next(iter(dataloader))
samples = samples.to(device)

y = net(samples)

print(y)