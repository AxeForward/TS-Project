import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.ts_tcc_loss import NTXentLoss
from utils import DataTransform



def Trainer(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device,num_epoch, logger,experiment_log_dir,training_mode,jitter_scale_ratio,jitter_ratio,max_seg,batch_size, temperature=0.2,use_cosine_similarity=True):
    # Start training
    #logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    for epoch in range(1, num_epoch + 1):
        # Train and validate
        train_loss, train_acc,dict = model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_dl, device, training_mode,
                                                jitter_scale_ratio,jitter_ratio,max_seg,batch_size, temperature,use_cosine_similarity)
        valid_loss, valid_acc, _, _ = model_evaluate(model, temporal_contr_model, valid_dl, device, training_mode)
        if training_mode != 'self_supervised':  # use scheduler in all other modes.
            scheduler.step(valid_loss)

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                     f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}')

    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(), 'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    if training_mode != "self_supervised":  # no need to run the evaluation for self-supervised mode.
        # evaluate on the test set
        logger.debug('\nEvaluate on the Test set:')
        test_loss, test_acc, _, _ = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
        logger.debug(f'Test loss      :{test_loss:0.4f}\t | Test Accuracy      : {test_acc:0.4f}')

    logger.debug("\n################## Training is Done! #########################")
    logger.debug(f'data shape :')
    for key,value in dict.items():
        logger.debug(f'{key:20}: {value}')



def model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_loader, device, training_mode,
                jitter_scale_ratio,jitter_ratio,max_seg,batch_size, temperature,use_cosine_similarity):
    total_loss = []
    total_acc = []
    model.train()
    temporal_contr_model.train()

    for batch_idx, (data, labels, aug1, aug2) in enumerate(train_loader):
        # send to device
        aug1,aug2 = DataTransform(data,jitter_scale_ratio,jitter_ratio,max_seg)
        data, labels = data.float().to(device), labels.long().to(device)
        aug1 = torch.from_numpy(aug1)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)
        # optimizer
        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()

        if training_mode == "self_supervised":
            predictions1, features1 = model(aug1)
            predictions2, features2 = model(aug2)
      
            # normalize projection feature vectors
            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)

            temp_cont_loss1, temp_cont_lstm_feat1,c_tokens1,fai_l1,fai_01,z_h1,c_t1,forward_seq1,encode_samples1,pred1= temporal_contr_model(features1, features2)
            temp_cont_loss2, temp_cont_lstm_feat2,c_tokens2,fai_l2,fai_02,z_h2,c_t2,forward_seq2,encode_samples,pred2 = temporal_contr_model(features2, features1)

            # normalize projection feature vectors
            zis = temp_cont_lstm_feat1 
            zjs = temp_cont_lstm_feat2 

        else:
            output = model(data)

        # compute loss
        if training_mode == "self_supervised":
            lambda1 = 1
            lambda2 = 0.7
            nt_xent_criterion = NTXentLoss(device, batch_size, temperature,
                                           use_cosine_similarity)
            loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 +  nt_xent_criterion(zis, zjs) * lambda2
            
        else: # supervised training or fine tuining
            predictions, features = output
            print(predictions.shape,labels.shape)
            loss = criterion(predictions, labels)
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        temp_cont_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()
    dict = {'sample_shape ': data.shape,
            'label_shape': labels.shape,
            'aug1_shape' : aug1.shape,
            # 'z_aug1_shape' : features1.shape,
            # 'forward_seq_shape' : forward_seq1.shape,
            # 'z_h_shape' : z_h1.shape,
            # 'c_token_shape' : c_tokens1.shape,
            # 'fai0_shape' : fai_01.shape,
            # 'faiL_shape' : fai_l1.shape,
            # 'c_t_shape': c_t1.shape,
            # 'ct_linear_shape':zis.shape,
            # 'encode_samples_shape' : encode_samples1.shape,
            # 'pred_shape': pred1.shape
            }

    if training_mode == "self_supervised":
        total_acc = 0
    else:
        total_acc = torch.tensor(total_acc).mean()
    return total_loss, total_acc,dict


def model_evaluate(model, temporal_contr_model, test_dl, device, training_mode):
    model.eval()
    temporal_contr_model.eval()

    total_loss = []
    total_acc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for data, labels, _, _ in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)

            if training_mode == "self_supervised":
                pass
            else:
                output = model(data)

            # compute loss
            if training_mode != "self_supervised":
                predictions, features = output
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                total_loss.append(loss.item())

            if training_mode != "self_supervised":
                pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

    if training_mode != "self_supervised":
        total_loss = torch.tensor(total_loss).mean()  # average loss
    else:
        total_loss = 0
    if training_mode == "self_supervised":
        total_acc = 0
        return total_loss, total_acc, [], []
    else:
        total_acc = torch.tensor(total_acc).mean()  # average acc
    return total_loss, total_acc, outs, trgs
