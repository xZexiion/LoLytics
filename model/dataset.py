import torch
import torch.utils.data as dutils
import pickle
import lmdb
import random

def load_data(data_path):
    random.seed(42)
    keys = []
    env = lmdb.open('dataset.lmdb', readonly=True, lock=False)
    with env.begin() as txn:
        with txn.cursor() as cursor:
            for key, _ in cursor:
                keys.append(key)
    random.shuffle(keys)

    return {
        'env': env,
        'keys': keys
    }

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