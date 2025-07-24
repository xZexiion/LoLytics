from dataset import Dataset, load_data
from dnn import DNN
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using {device}')

net = DNN()
net = net.to(device)

data = load_data('dataset.lmdb')
train_ds = Dataset(data['keys'], data['env'], True, 0.8)
test_ds = Dataset(data['keys'], data['env'], False, 0.8)
train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=512, shuffle=True)

optimizer = optim.Adam(net.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

def train_epoch(model, optimizer, criterion, dataloader):
    model.train()
    avg_loss = 0

    for inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        y_pred = model(inputs)
        loss = criterion(y_pred, labels.float().view(-1, 1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()

    avg_loss /= len(dataloader)

    return avg_loss

def test(model, optimizer, criterion, dataloader):
    model.train()
    avg_loss = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            y_pred = model(inputs)
            loss = criterion(y_pred, labels.float().view(-1, 1))

            avg_loss += loss.item()

        avg_loss /= len(dataloader)

    return avg_loss

train_losses = []
test_losses = []
for epoch in range(10):
    train_loss = train_epoch(net, optimizer, loss_fn, train_dl)
    test_loss = test(net, optimizer, loss_fn, test_dl)
    print(f'Epoch {epoch+1}) Train loss: {train_loss} Test loss: {test_loss}')

plt.figure(figsize=(8, 8))
plt.title('Training And Testing Loss')
plt.plot(train_losses, label='Train')
plt.plot(test_losses, label='Test')
plt.legend()
plt.show()


torch.save(net.state_dict(), 'net.pth')
