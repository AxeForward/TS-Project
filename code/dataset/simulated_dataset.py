import torch
import numpy as np
from torch.utils.data import Dataset
import os
import pickle

class simulateddataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, x_data, y_data):
        super(simulateddataset, self).__init__()
        if len(x_data.shape) < 3:
            x_data = x_data.unsqueeze(2)

        if x_data.shape.index(min(x_data.shape)) != 1:  # make sure the Channels in second dim
            x_data = x_data.permute(0, 2, 1)

        if isinstance(x_data, np.ndarray):
            self.x_data = torch.from_numpy(x_data)
            self.y_data = torch.from_numpy(y_data).long()
        else:
            self.x_data = x_data
            self.y_data = y_data

        self.len = x_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.x_data[index], self.x_data[index]

    def __len__(self):
        return self.len
