import glob
import importlib
import sys,os
#BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append("..")

import pandas as pd

def get_raw_data(data_source_name,selected_data_name):#UCI_Epilepsy
 
    dataset_module = importlib.import_module('dataset.'+str(data_source_name.lower())+'_dataset')
    if data_source_name == 'UCI':
        if selected_data_name == 'Epilepsy':
            dataset = pd.read_csv(r'E:\ts_proj\TS-Project\data\UCI\Epilepsy\data.csv')
            train_data,val_data,test_data = dataset_module.reprocess_epil(dataset)
        elif selected_data_name == 'HAR':
            print(1)
    elif data_source_name == 'Kaggle':
        all_trials_folders = sorted(glob.glob(r'E:\ts_proj\TS-Project\data\Kaggle\\' +selected_data_name+ "/*"))
        train_data, val_data,test_data = dataset_module.reprocess_motion(all_trials_folders,window_size=400)
    elif data_source_name == 'UEA':
        print(1)
    #print(train_data['samples'].shape,train_data['labels'].shape)
    #print(train_data[0].shape,train_data[1].shape)
    return train_data,val_data,test_data

