import torch
from dataset import Dataset
from dnn import DNN
from tqdm import tqdm
import matplotlib.pyplot as plt

dataset = Dataset('test.lmdb')

data_iter = iter(dataset)

l = [0] * 40
t = [0] * 40

net = DNN()
net.load_state_dict(torch.load('dnn.pth', weights_only=True))

with torch.no_grad():
    for sample in tqdm(dataset):
        inputs, label = sample

        inputs = torch.tensor(inputs).view(1, -1)
        label = torch.tensor(label, dtype=torch.float)

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
