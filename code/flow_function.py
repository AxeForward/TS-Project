import torch as th
import os
import sys
from dataset.ucr_dataset import UCRDataset
from dataset.uea_dataset import UEADataset
#from dataset.physionet_dataset import Physionet_Dataset
#from dataset.cardiology_dataset import Cardiology_Dataset
from models.mix_up import FCN 
from trainers.train_mix_up import train_mixup_model_epoch
from utils import set_global_seed
from load_data.load_ucr import load_ucr
from load_data.load_uea import load_uea
#from load_data.load_physionet import load_physionet2017_data, load_physionet2020_data
#from load_data.load_cardiology import load_cardiology_data
import pandas as pd

def flow_mixup_function(data_source:str, filename:str, epochs:int, batch_size:int,
                        alpha:float, random_seed:int, device:str='cpu'):

    set_global_seed(random_seed)
    if data_source == 'ucr':
        x_tr, y_tr, x_te, y_te = load_ucr(filename=filename)

        training_set = UCRDataset(x_tr, y_tr)
        test_set = UCRDataset(x_te, y_te)
    elif data_source == 'uea':
        x_tr, y_tr, x_te, y_te = load_uea(filename=filename)

        training_set = UEADataset(x_tr, y_tr)
        test_set = UEADataset(x_te, y_te)
    elif data_source == 'physionet2017':
        filename = 'physionet2017'
        x_tr, y_tr, x_te, y_te = load_physionet2017_data()

        training_set = Physionet_Dataset( x_tr, y_tr)
        test_set = Physionet_Dataset(x_te, y_te)
    elif data_source == 'physionet2020':
        filename = 'physionet2020'
        x_tr, y_tr, x_te, y_te = load_physionet2020_data()

        training_set = Physionet_Dataset( x_tr, y_tr)
        test_set = Physionet_Dataset(x_te, y_te)
    elif data_source == 'cardiology':
        filename = 'cardiology'
        x_tr, y_tr, x_te, y_te = load_cardiology_data()

        training_set = Cardiology_Dataset( x_tr, y_tr)
        test_set = Cardiology_Dataset(x_te, y_te)
    else:
        print('Please enter our supported data source')
        sys.exit()
   
    model = FCN(training_set.x.shape[1]).to(device)
    optimizer = th.optim.Adam(model.parameters())
    LossListM, AccListM = train_mixup_model_epoch(model, training_set, test_set,
                                              optimizer, alpha, epochs, batch_size, device)

    '''
    print(f"Score for alpha = {alpha}: {AccListM[-1]}")

    model_result = pd.DataFrame({"loss": LossListM, "acc": AccListM})
    res_file_name = filename + '_' + str(epochs) + 'epochs' + '_' + str(random_seed) + 'seed' + '.csv'
    res_path = os.path.join(os.path.dirname(__file__), 'tests', 'results', 'mix_up', data_source.upper(), res_file_name)
    model_result.to_csv(res_path)
    '''
    return LossListM, AccListM

if __name__ == '__main__':
    print(flow_mixup_function('ucr', 'GunPoint', 10, 50, 1.0, 42))

