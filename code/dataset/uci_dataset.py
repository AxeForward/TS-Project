import sys
sys.path.append("..")
from utils import to_dict
import torch
from torch.utils.data import Dataset
import numpy as np

def reprocess_epil(dataset_epil):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    y = dataset_epil.iloc[:, -1]
    x = dataset_epil.iloc[:, 1:-1]

    x = x.to_numpy()
    y = y.to_numpy()
    y = y - 1
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    for i, j in enumerate(y):
        if j != 0:
            y[i] = 1

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    names = locals()
    
    return to_dict(names)



class My_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset):
        super(My_Dataset, self).__init__()
        #self.training_mode = training_mode

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if isinstance(X_train, np.ndarray):
            X_train = torch.from_numpy(X_train)
            y_train = torch.from_numpy(y_train).long()

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)#升维

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        self.x_data = X_train
        self.y_data = y_train
        self.len = X_train.shape[0]


    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.x_data[index], self.x_data[index]

    def __len__(self):
        return self.len
