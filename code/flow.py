# If tests are needed, just cancel one of those comments
# Main Running File

import torch as th
import argparse
import os
import sys
from sktime.datasets import load_gunpoint
import dataset, models, load_data, trainers, utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasource', type=str, default='uea', help='ucr or uea datasets')
    parser.add_argument('--dataset', type=str, default='ERing', help='The dataset name')
    parser.add_argument('--epochs', type=int, default=10, help='Epoch number')
    parser.add_argument('--alpha', type=float, default=1.0, help='The alpha')
    parser.add_argument('--device', type=str, default='cpu' ,help='cpu for CPU and cuda for NVIDIA GPU')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
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

    utils.set_global_seed(random_seed)
    if data_source == 'ucr':
        pass
    elif data_source == 'uea':
        pass
    elif data_source == 'kpi':
        flow_dir = os.path.dirname(__file__)
        project_dir = os.path.dirname(flow_dir)
        data_path = os.path.join(project_dir, 'data', 'KPI', 'kpi.pkl')
        dict_train_data, dict_train_labels, dict_train_timestamps, \
            dict_test_data, dict_test_labels, dict_test_timestamps = dataset.kpi_dataset.load_anomaly(data_path)

        training_dataset = dataset.kpi_dataset.KPIdataset(dict_train_data, dict_train_labels, dict_train_timestamps).data_set
    else:
        print('Please enter our supported data source')
        sys.exit()
   
    model = models.mix_up.FCN(training_dataset.shape[1]).to(device)
    optimizer = th.optim.Adam(model.parameters())
    LossListM, AccListM = train_mixup_model_epoch(model, training_set, test_set,
                                              optimizer, alpha, epochs)

    #print(LossListM)

    print(f"Score for alpha = {alpha}: {AccListM[-1]}")

else:
    print('Someting Wrong')







'''
import torch
import numpy as np
from datetime import datetime
import argparse
import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)
import importlib
from load_data import load_raw
import utils

parser = argparse.ArgumentParser()

######################## Model parameters ########################

parser.add_argument('--source_dataset', default='UCI_Epilepsy', type=str,help='')
parser.add_argument('--batch_size', default=128, type=int,help='')    
parser.add_argument('--epoch', default=5, type=int,help='')                
parser.add_argument('--seed', default=0, type=int,help='seed value')
parser.add_argument('--model', default='ts_tcc', type=str,help='')
parser.add_argument('--training_mode', default='self_supervised', type=str,help='Modes of choice: random_init, supervised, self_supervised, fine_tune, train_linear')
parser.add_argument('--input_channels', default=1, type=int,help='input_channels')
parser.add_argument('--kernel_size',default=8,type=int,help='')
parser.add_argument('--stride',default=1,type=int, help='')
parser.add_argument('--dropout',default=0.35,type=int,help='')
parser.add_argument('--final_out_channels',default=128,type=int,help='')
parser.add_argument('--features_len',default=24,type=int,help='')                    
parser.add_argument('--num_classes',default=2,type=int,help='')   
parser.add_argument('--TC_timesteps',default=10,type=int,help='') 
parser.add_argument('--TC_hidden_dim',default=100,type=int,help='')
parser.add_argument('--jitter_scale_ratio',default=0.001,type=float,help='')
parser.add_argument('--jitter_ratio',default=0.001,type=float,help='')
parser.add_argument('--max_seg',default=5,type=int,help='')

args = parser.parse_args()

device = 'cpu'
dataset_name = args.source_dataset
batch_size = args.batch_size
num_epoch = args.epoch
model_name = args.model
training_mode = args.training_mode
input_channels = args.input_channels
kernel_size = args.kernel_size
stride = args.stride
dropout = args.dropout
final_out_channels = args.final_out_channels
features_len = args.features_len
num_classes = args.num_classes
TC_timesteps = args.TC_timesteps
TC_hidden_dim = args.TC_hidden_dim
jitter_scale_ratio = args.jitter_scale_ratio
jitter_ratio = args.jitter_ratio
max_seg = args.max_seg

[data_source_name,selected_data_name] = dataset_name.split('_')
Dataset = importlib.import_module('dataset.'+str(data_source_name.lower())+'_dataset')
model_import = importlib.import_module('models.'+str(model_name))
train_model = importlib.import_module('trainers.train_'+str(model_name))
print('##########_load_data_############')
#####################load datasets#####################################################
train_data,val_data,test_data = load_raw.get_raw_data(data_source_name,selected_data_name)
train_dataset = Dataset.My_Dataset(train_data)
valid_dataset = Dataset.My_Dataset(val_data)
test_dataset = Dataset.My_Dataset(test_data)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                            shuffle=True, drop_last=True,
                                            num_workers=0)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size,
                                            shuffle=False, drop_last=True,
                                            num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                            shuffle=False, drop_last=True,
                                            num_workers=0)
print("##########_succeed_load_#########")
# ##### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################
if model_name == 'ts_tcc':
    experiment_log_dir = os.path.join('../experiment_description', model_name,selected_data_name)
    print(experiment_log_dir)
    os.makedirs(experiment_log_dir, exist_ok=True)
    # Logging ######################################
    log_file_name = os.path.join(experiment_log_dir, training_mode + f"_seed_{SEED}.log")
    logger = utils._logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Dataset: {selected_data_name}')
    logger.debug(f'Method:  TS-TCC')
    logger.debug(f'Mode:    {training_mode}')
    logger.debug("=" * 45)
    #################################################
    model = model_import.base_Model(input_channels,final_out_channels,features_len,num_classes,kernel_size,stride,dropout).to(device)
    temporal_contr_model = model_import.TC(final_out_channels,TC_timesteps,TC_hidden_dim, device).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.99), weight_decay=3e-4)
    temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=3e-4, betas=(0.9, 0.99), weight_decay=3e-4)
    if training_mode == 'fine_tune':
        model = model_import.get_fine_tune_model(model,experiment_log_dir,device)
    print('##########_training_start_###########')
    train_model.Trainer(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_loader, valid_loader, test_loader, 
                        device,num_epoch, logger,experiment_log_dir,training_mode,jitter_scale_ratio,jitter_ratio,max_seg,batch_size, temperature=0.2,use_cosine_similarity=True)
    print('###########_training_done_############')

    if training_mode != "self_supervised":
    # Testing
        outs = train_model.model_evaluate(model, temporal_contr_model, test_loader, device, training_mode)
        total_loss, total_acc, pred_labels, true_labels = outs
        utils._calc_metrics(pred_labels, true_labels, experiment_log_dir)

elif model_name == 'mix_up':
    print(1)
'''


