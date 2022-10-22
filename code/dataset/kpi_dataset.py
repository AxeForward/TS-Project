import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset
import typing

def pkl_save(name, var):
    with open(name, 'wb') as f:
        pickle.dump(var, f)

def pkl_load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def load_anomaly(data_path):
    res = pkl_load(data_path)
    return res['all_train_data'], res['all_train_labels'], res['all_train_timestamps'], \
           res['all_test_data'],  res['all_test_labels'],  res['all_test_timestamps']


def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size//2)
    else:
        npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=np.nan)

def reconstruct_label(timestamp, label):
    timestamp = np.asarray(timestamp, np.int64)
    index = np.argsort(timestamp)

    timestamp_sorted = np.asarray(timestamp[index])
    interval = np.min(np.diff(timestamp_sorted))

    label = np.asarray(label, np.int64)
    label = np.asarray(label[index])

    idx = (timestamp_sorted - timestamp_sorted[0]) // interval

    new_label = np.zeros(shape=((timestamp_sorted[-1] - timestamp_sorted[0]) // interval + 1,), dtype=np.int)
    new_label[idx] = label

    return new_label

def gen_ano_array_data(load_data):
    maxl = np.max([ len(load_data[k]) for k in load_data ])
    pretrain_data = []
    for k in load_data:
        filled_data = pad_nan_to_target(load_data[k], maxl, axis=0)
        pretrain_data.append(filled_data)
    pretrain_data = np.expand_dims(np.stack(pretrain_data), 2)
    origin_shape = pretrain_data.shape
    pretrain_data = pretrain_data.reshape(origin_shape[0], origin_shape[2], origin_shape[1])
    return pretrain_data

def gen_ano_array_labels(load_data, load_labels, load_timestamps):
    labels_log = []
    timestamps_log = []

    for k in load_data:
        array_labels = load_labels[k]
        array_timestamps = load_timestamps[k]
        labels_log.append(array_labels)
        timestamps_log.append(array_timestamps)

    new_labels = []

    for array_labels_k, array_timestamps_k in zip(labels_log, timestamps_log):
        assert len(labels_log) == len(timestamps_log)
        array_labels_after = reconstruct_label(array_timestamps_k, array_labels_k)
        new_labels.append(array_labels_after)

    return new_labels

class KPIdataset(Dataset):
    def __init__(self, load_data:typing.Dict, load_labels:typing.Dict, load_timestamps:typing.Dict, device:str='cpu'):
        self.data_set = gen_ano_array_data(load_data)
        self.data_labels = gen_ano_array_labels(load_data, load_labels, load_timestamps)
        self.data_set = torch.tensor(self.data_set, dtype=torch.float, device=device)
        self.data_labels = [torch.tensor(i, dtype=torch.long, device=device) for i in self.data_labels]

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        return self.data_set[idx], self.data_labels[idx]


    
