import torch
from dataset import Dataset, load_data
from dnn import DNN
from tqdm import tqdm
import matplotlib.pyplot as plt

data = load_data('dataset.lmdb')
dataset = Dataset(data['keys'], data['env'], False, 0.8)

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using {device}')

data_iter = iter(dataset)

l = [0] * 40
t = [0] * 40


net = DNN()
net.load_state_dict(torch.load('net.pth', weights_only=True))
net = net.to(device)

with torch.no_grad():
    for _ in tqdm(range(100000)):
        sample = next(data_iter)
        inputs, label = sample

        time = inputs[-1]
        if time >= 40:
            continue
        inputs = torch.tensor(inputs).view(1, -1).to(device)
        y = torch.sigmoid(net(inputs)).round()

        l[time] += (y == torch.tensor(label).to(device)).float().item()
        t[time] += 1

for i in range(40):
    l[i] /= t[i]

x = list(range(40))

plt.figure(figsize=(8, 8))
plt.bar(x, l)
plt.show()
