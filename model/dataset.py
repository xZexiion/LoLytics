import torch
import torch.utils.data as dutils
import lmdb
import pickle

class Dataset(dutils.Dataset):
    def __init__(self, data_path, training, split):
        keys = []
        self.env = lmdb.open(data_path, readonly=True, lock=False)
        with self.env.begin() as txn:
            with txn.cursor() as cursor:
                for key, _ in cursor:
                    keys.append(key)
        keys.sort()
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
        return sample

if __name__ == '__main__':
    ds = Dataset('dataset.lmdb', False, 0.8)
    print(len(ds))
    a = next(iter(ds))
    print(a)