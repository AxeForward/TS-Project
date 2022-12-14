import pandas as pd
import numpy as np, os
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

def load_physionet2020_data():
    directory_path = r'./data/PhysioNet2020/Training_WFDB/'
    x = list()
    y = list()
    for file in os.listdir(directory_path)[:100]: ###################
        file_path = os.path.join(directory_path, file)
        if not file.lower().startswith('.') and file.lower().endswith('mat') and os.path.isfile(file_path):
            data = loadmat(file_path)
            data_x = np.asarray(data['val'], dtype=np.float64) # data_x.shape: 12*length
            data_x = data_x[0:12,0:4000] #############
            data_x = data_x.tolist()
            x.append(data_x)
            filehea_path = file_path.replace('.mat','.hea')
            with open(filehea_path, 'r') as f:
                for line in f:
                    if line.startswith('#Dx'):
                        tmp = line.split(': ')[1].split(',')
                        for c in tmp:
                            y.append(c)
                            break
    y = shift_y(y)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    return X_train, y_train, X_test, y_test

def load_physionet2017_data():
    training_path = r'./data/PhysioNet2017/training2017/'
    label_path = r'./data/PhysioNet2017/label/REFERENCE.csv'
    files = os.listdir(training_path)
    filenames = [file for file in files if file.endswith('.mat')]
    x = []
    for name in filenames[:10]:###########
        data = loadmat(os.path.join(training_path,name))
        data_x = np.asarray(data['val'], dtype=np.float64) 
        data_x = data_x.tolist()[0]
        data_x = data_x[:9000] #############
        x.append(data_x)
    label_data = pd.read_csv(label_path, header=None)
    label_data = label_data.iloc[:10,:]###########
    y = shift_y( list(label_data.iloc[:,1]) )
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    X_train, X_test = pd.DataFrame({'x':X_train}), pd.DataFrame({'x':X_test})
    return X_train, y_train, X_test, y_test

def shift_y(x):
    new = []
    uniq = list(set(x))
    for i in x:
        n = 0
        for j in uniq:
            if i == j:
                new.append(n)
            else:
                n += 1
    return new