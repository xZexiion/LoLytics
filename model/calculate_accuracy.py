import torch
from dataset import Dataset
from dnn import DNN
from tqdm import tqdm
import matplotlib.pyplot as plt

dataset = Dataset('test.lmdb')

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using {device}')

data_iter = iter(dataset)

l = [0] * 40
t = [0] * 40


net = DNN()
net.load_state_dict(torch.load('net.pth', map_location=device, weights_only=True))
# net = net.to(device)

with torch.no_grad():
    for _ in tqdm(range(100000)):
        sample = next(data_iter)
        inputs, label = sample

        time = inputs[-1]
        if time >= 40:
            continue
        inputs = torch.tensor(inputs).view(1, -1)
        y = torch.sigmoid(net(inputs)).round()

        l[time] += (y == torch.tensor(label, dtype=torch.float)).item()
        t[time] += 1

for i in range(40):
    l[i] /= t[i]

x = list(range(40))

plt.figure(figsize=(5, 5))
plt.bar(x, l)
plt.xlabel('Time Interval')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Over Time')
plt.show()
