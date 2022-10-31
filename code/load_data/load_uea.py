from enum import unique
import pandas as pd
import numpy as np
import os
from scipy.io.arff import loadarff

# shift label from string to int
def shift_label(label:list):
    label_shifted = []
    uniq = list(set(label))
    for i in label:
        n = 0
        for j in uniq:
            if i == j:
                label_shifted.append(n)
            else:
                n += 1
    return label_shifted

def uea_process(filename: str, type: str):
    path = r'TS-Project\data\UEA\Epilepsy\{}_{}.arff'.format(filename,type.upper())
    with open(path, 'r') as file:
        file_content = file.readlines()

    rows = []
    for lines in range(len(file_content)):
        if file_content[lines].startswith('@data'):
            start = lines
            break
    rows = file_content[start+1:]
    data_x = [row[1:row.rfind(',')-1] for row in rows]
    data_x = [row.split('\\n') for row in data_x]
    for row in range(len(data_x)):
        data_x[row] = [list(eval(factor)) for factor in data_x[row]]
    data_x = pd.DataFrame(data_x)
    
    data_y = [row[row.rfind(',')+1:] for row in rows]
    data_y = shift_label(data_y)
    data_y = pd.DataFrame(data_y)

def load_uea(filename: str):
    train_x, train_y = uea_process(filename, 'train')
    test_x, test_y = uea_process(filename, 'test')
    return train_x, train_y, test_x, test_y
