import torch
import torch.utils.data as dutils
import pickle
import lmdb

class Dataset(dutils.Dataset):
    def __init__(self, data_path):
        self.keys = []
        self.env = lmdb.open(data_path, readonly=True, lock=False)
        with self.env.begin() as txn:
            with txn.cursor() as cursor:
                for key, _ in cursor:
                    self.keys.append(key)

        print(f'Found {len(self.keys)} samples in {data_path}')
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.env.begin() as txn:
            data = txn.get(key)
        sample = pickle.loads(data)

        return sample[:-1], sample[-1]

if __name__ == '__main__':
    ds = Dataset('dataset.lmdb')
    print(len(ds))
    a = next(iter(ds))
    print(a)