# Train and evlueate the model
import torch as th
import torch.nn as nn
import numpy as np
from IPython.display import clear_output
import os
from torch.utils.data import Dataset, DataLoader
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from sklearn.neighbors import KNeighborsClassifier
from utils import to_np
from losses.mix_up_loss import MixUpLoss

def test_model(model, training_set, test_set):

    model.eval()

    N_tr = len(training_set.x)
    N_te = len(test_set.x)

    training_generator = DataLoader(training_set, batch_size=1,
                                    shuffle=True, drop_last=False)
    test_generator = DataLoader(test_set, batch_size= 1,
                                    shuffle=True, drop_last=False)

    H_tr = th.zeros((N_tr, 128))
    y_tr = th.zeros((N_tr), dtype=th.long)

    H_te = th.zeros((N_te, 128))
    y_te = th.zeros((N_te), dtype=th.long)

    for idx_tr, (x_tr, y_tr_i) in enumerate(training_generator):
        with th.no_grad():
            _, H_tr_i = model(x_tr)
            H_tr[idx_tr] = H_tr_i
            y_tr[idx_tr] = y_tr_i

    H_tr = to_np(nn.functional.normalize(H_tr))
    y_tr = to_np(y_tr)


    for idx_te, (x_te, y_te_i) in enumerate(test_generator):
        with th.no_grad():
            _, H_te_i = model(x_te)
            H_te[idx_te] = H_te_i
            y_te[idx_te] = y_te_i

    H_te = to_np(nn.functional.normalize(H_te))
    y_te = to_np(y_te)

    clf = KNeighborsClassifier(n_neighbors=1).fit(H_tr, y_tr)

    return clf.score(H_te, y_te)


def train_mixup_model_epoch(model, training_set, test_set, optimizer, alpha, epochs):

    device = 'cuda' if th.cuda.is_available() else 'cpu'
    batch_size_tr = len(training_set.x)

    LossList, AccList = [] , []
    criterion = MixUpLoss(device, batch_size_tr)

    training_generator = DataLoader(training_set, batch_size=batch_size_tr,
                                    shuffle=True, drop_last=True)

    for epoch in range(epochs):

        for x, y in training_generator:

            model.train()

            optimizer.zero_grad()

            x_1 = x
            x_2 = x[th.randperm(len(x))]

            lam = np.random.beta(alpha, alpha)

            x_aug = lam * x_1 + (1-lam) * x_2

            z_1, _ = model(x_1)
            z_2, _ = model(x_2)
            z_aug, _ = model(x_aug)

            loss= criterion(z_aug, z_1, z_2, lam)
            loss.backward()
            optimizer.step()
            LossList.append(loss.item())


        AccList.append(test_model(model, training_set, test_set))

        print(f"Epoch number: {epoch}")
        print(f"Loss: {LossList[-1]}")
        print(f"Accuracy: {AccList[-1]}")
        print("-"*50)

        if epoch % 10 == 0 and epoch != 0: clear_output() # May fail
        #if epoch % 10 == 0 and epoch != 0:
        #    os.system('cls')
            
    return LossList, AccList