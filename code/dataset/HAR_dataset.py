import torch
import numpy as np
from torch.utils.data import Dataset
import os
from typing import Optional
data_path = r'../data/UCI/HAR'
from sklearn.model_selection import train_test_split

class HARdataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset):
        super(HARdataset, self).__init__()

        x_train = dataset["samples"]
        y_train = dataset["labels"]

        if len(x_train.shape) < 3:
            x_train = x_train.unsqueeze(2)

        if x_train.shape.index(min(x_train.shape)) != 1:  # make sure the Channels in second dim
            x_train = x_train.permute(0, 2, 1)

        if isinstance(x_train, np.ndarray):
            self.x_data = torch.from_numpy(x_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = x_train
            self.y_data = y_train

        self.len = x_train.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.x_data[index], self.x_data[index]

    def __len__(self):
        return self.len

def get_HAR(data_path:str):
    if os.access(os.path.join(data_path, "train.pt"),os.R_OK) == False:
        train_data = load_HAR_data('train')
        train_label = get_HAR_label('train')
        x_test = load_HAR_data('test')
        y_test = get_HAR_label('test')
        x_train, x_val, y_train, y_val = train_test_split(train_data, train_label, test_size=0.2, random_state=42)
        save_HAR_data(x_train, y_train, 'train.pt')
        save_HAR_data(x_val, y_val, 'val.pt')
        save_HAR_data(x_test, y_test, 'test.pt')

def load_HAR_data(stage: Optional[str] = 'train') -> np.ndarray:

    data_dir = 'UCI HAR Dataset'
    root_path = os.path.join(stage, 'Inertial Signals')
    data_list = ['body_acc', 'body_gyro', 'total_acc']
    state = ['x', 'y', 'z']

    file_name = ['{0}_{1}_{2}.txt'.format(i, j, stage) for i in data_list for j in state]
    file_path = [os.path.join(data_dir, root_path, file) for file in file_name]

    data_sets = (np.loadtxt(file) for file in file_path)

    return np.stack(data_sets, axis=1)

def get_HAR_label(stage: Optional[str] = 'train') -> np.ndarray:

    data_dir = 'UCI HAR Dataset'
    file_name = 'y_{}.txt'.format(stage)
    file_path = os.path.join(data_dir, stage, file_name)
    data_label = np.loadtxt(file_path)
    data_label -= np.min(data_label)

    return data_label

def save_HAR_data(data: np.ndarray, label: np.ndarray, file_name: str, out_dir: Optional[str] = './data/HAR'):

    data_dict = dict()
    data_dict['samples'] = torch.from_numpy(data)
    data_dict['labels'] = torch.from_numpy(label)
    torch.save(data_dict, os.path.join(out_dir, file_name))
