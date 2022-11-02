#@title Load packages and data


import torch as th
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


#@title load data and convert to numpy array

def toarray(x,y):
    x = pd.DataFrame(x).to_numpy()
    y = pd.DataFrame(y).to_numpy()

    x = np.array(np.ndarray.tolist(x), dtype=np.float32)
    y = np.array(np.ndarray.tolist(y), dtype=np.int32)
    return x,y


#@title create dataset

class MyDataset(Dataset):
    def __init__(self, x, y):

        device = 'cuda' if th.cuda.is_available() else 'cpu'
        self.x1, self.y1 = toarray(x, y)
        self.x = th.tensor(self.x1, dtype=th.float, device=device)
        self.y = th.tensor(self.y1, dtype=th.long, device=device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]