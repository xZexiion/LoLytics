import torch
import torch.utils.data as dutils
import pickle

class Dataset(dutils.Dataset):
    def __init__(self, keys, env, training, split):
        self.env = env
        num_train_samples = int(len(keys) * split)
        if training:
            self.keys = keys[:num_train_samples]
        else:
            self.keys = keys[num_train_samples:]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.env.begin() as txn:
            data = txn.get(key)
        sample = pickle.loads(data)

        return sample[:-1], sample[-1]

if __name__ == '__main__':
    ds = Dataset('dataset.lmdb', False, 0.8)
    print(len(ds))
    a = next(iter(ds))
    print(a)