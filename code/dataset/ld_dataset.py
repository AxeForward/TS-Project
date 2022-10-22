import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import typing

def gen_array_data(file_path):
    data_read = pd.read_csv(file_path, index_col='date', parse_dates=True)
    data_load = data_read.to_numpy()
    return data_load

class LDdataset(Dataset):
    def __init__(self, data_load:np.array, partial_data:str='train', device:str='cpu'):
        self.train_slice = slice(None, int(0.6 * len(data_load)))
        self.valid_slice = slice(int(0.6 * len(data_load)), int(0.8 * len(data_load)))
        self.test_slice = slice(int(0.8 * len(data_load)), None)
        scaler = StandardScaler().fit(data_load[self.train_slice])
        self.all_data = scaler.transform(data_load)

        if partial_data == 'train':
            self.data_set = self.all_data[self.train_slice]
        elif partial_data == 'test':
            self.data_set = self.all_data[self.test_slice]
        elif partial_data == 'valid':
            self.data_set = self.all_data[self.valid_slice]
        else:
            print('Please Type Correct Request.')
            self.data_set = torch.zeros(self.all_data.shape)
    
        self.data_set = np.expand_dims(self.data_set.T, 1)
        self.data_set = torch.tensor(self.data_set, dtype=torch.float, device=device)
    
    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        return self.data_set[idx]


