import pandas as pd

def ucr_process(filename: str, type: str ):
    path = 'data/UCR/{}/{}_{}.tsv'.format(filename, filename, type.upper())
    data = pd.read_csv(path, sep='\t', header=None)
    data = pd.DataFrame(data)
    data_x = data.iloc[:,1:]
    data_x_list = []
    for rows in range(data_x.shape[0]):
        data_x_list.append(list(data_x.iloc[rows,:]))
    data_x.iloc[:,0] = data_x_list
    data_x = data_x.iloc[:,0]
    data_y = data.iloc[:,0]
    data_x = pd.DataFrame(data_x)
    data_y = pd.DataFrame(data_y)
    return data_x, data_y

def load_ucr(filename: str):
    train_x, train_y = ucr_process(filename, 'train')
    test_x, test_y = ucr_process(filename, 'test')
    return train_x, train_y, test_x, test_y


