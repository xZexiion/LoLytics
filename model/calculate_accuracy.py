import torch
from dataset import Dataset
from dnn import DNN
from tqdm import tqdm
import matplotlib.pyplot as plt

dataset = Dataset('test.lmdb')

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
device = 'cpu'
print(f'Using {device}')

data_iter = iter(dataset)

l = [0] * 40
t = [0] * 40

net = DNN()
net.load_state_dict(torch.load('dnn.pth', map_location=device, weights_only=True))
net = net.to(device)

with torch.no_grad():
    for sample in tqdm(dataset):
        inputs, label = sample

        inputs = torch.tensor(inputs).view(1, -1).to(device)
        label = torch.tensor(label, dtype=torch.float).to(device)

        time = inputs[0][-1].item()
        if time >= 40:
            continue
        y = torch.sigmoid(net(inputs)).round()

        l[time] += (y == label).float().item()
        t[time] += 1

for i in range(40):
    l[i] /= t[i]
    l[i] *= 100
    l[i] = round(l[i])

x = list(range(40))

plt.figure(figsize=(8, 5))
plt.bar(x, l)
plt.xlabel('Time Interval')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Over Time')
plt.yticks(list(range(0, 100, 10)))
plt.show()
