# Train and evlueate the model
import torch as th
import numpy as np
from IPython.display import clear_output
import os
from torch.utils.data import DataLoader
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils import to_np
from losses.mix_up_loss import MixUpLoss
from tasks.evaluate_mixup import evaluate_mixup_model


def train_mixup_model_epoch(model, training_set, test_set, optimizer, alpha, epochs, batch_size_tr):

    device = 'cuda' if th.cuda.is_available() else 'cpu'
    # batch_size_tr = len(training_set.x)

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


        AccList.append(evaluate_mixup_model(model, training_set, test_set))

        print(f"Epoch number: {epoch}")
        print(f"Loss: {LossList[-1]}")
        print(f"Accuracy: {AccList[-1]}")
        print("-"*50)

        if epoch % 10 == 0 and epoch != 0: clear_output() # May fail
        #if epoch % 10 == 0 and epoch != 0:
        #    os.system('cls')
            
    return LossList, AccList