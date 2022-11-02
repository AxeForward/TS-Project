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










