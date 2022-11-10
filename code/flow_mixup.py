# Main Running File
import torch as th
import argparse
import os
import sys
from dataset.ucr_dataset import UCRDataset
from dataset.uea_dataset import UEADataset
from models.mix_up import FCN 
from trainers.train_mix_up import train_mixup_model_epoch
from utils import set_global_seed
from load_data.load_ucr import load_ucr
from load_data.load_uea import load_uea
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasource', type=str, default='ucr', help='ucr or uea datasets')
    parser.add_argument('--dataset', type=str, default='ACSF1', help='The dataset name')
    parser.add_argument('--epochs', type=int, default=10, help='Epoch number')
    parser.add_argument('--alpha', type=float, default=1.0, help='The alpha')
    parser.add_argument('--device', type=str, default='cpu' ,help='cpu for CPU and cuda for NVIDIA GPU')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--batchsize_tr', type=int, default=10, help='batch_size_tr')
    
    args = parser.parse_args()
    print("Data Source:", args.datasource.upper())
    print("Dataset:", args.dataset)
    print("Arguments:", str(args))

    data_source = args.datasource
    device = args.device
    epochs = args.epochs
    alpha = args.alpha
    filename = args.dataset
    random_seed = args.seed
    batch_size_tr = args.batchsize_tr

    set_global_seed(random_seed)
    if data_source == 'ucr':
        x_tr, y_tr, x_te, y_te = load_ucr(filename=filename)

        training_set = UCRDataset(x_tr, y_tr)
        test_set = UCRDataset(x_te, y_te)
    elif data_source == 'uea':
        x_tr, y_tr, x_te, y_te = load_uea(filename=filename)

        training_set = UEADataset(x_tr, y_tr)
        test_set = UEADataset(x_te, y_te)
    else:
        print('Please enter our supported data source')
        sys.exit()
   
    model = FCN(training_set.x.shape[1]).to(device)
    optimizer = th.optim.Adam(model.parameters())
    LossListM, AccListM = train_mixup_model_epoch(model, training_set, test_set,
                                              optimizer, alpha, epochs, batch_size_tr)

    #print(LossListM)

    print(f"Score for alpha = {alpha}: {AccListM[-1]}")

    model_result = pd.DataFrame({"loss": LossListM, "acc": AccListM})
    res_file_name = args.dataset + '_' + str(epochs) + 'epochs' + '_' + str(random_seed) + 'seed' + '.csv'
    res_path = os.path.join(os.path.dirname(__file__), 'tests', 'results', 'mix_up', data_source.upper(), res_file_name)
    model_result.to_csv(res_path)

else:
    print('Someting Wrong')





