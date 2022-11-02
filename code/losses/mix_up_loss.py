# Loss Function

import torch as th
import torch.nn as nn

class MixUpLoss(th.nn.Module):

    def __init__(self, device, batch_size):
        super(MixUpLoss, self).__init__()
        
        self.tau = 0.5
        self.device = device
        self.batch_size = batch_size
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, z_aug, z_1, z_2, lam):

        z_1 = nn.functional.normalize(z_1)
        z_2 = nn.functional.normalize(z_2)
        z_aug = nn.functional.normalize(z_aug)

        labels_lam_0 = lam*th.eye(self.batch_size, device=self.device)
        labels_lam_1 = (1-lam)*th.eye(self.batch_size, device=self.device)

        labels = th.cat((labels_lam_0, labels_lam_1), 1)

        logits = th.cat((th.mm(z_aug, z_1.T),
                         th.mm(z_aug, z_2.T)), 1)

        loss = self.cross_entropy(logits / self.tau, labels)

        return loss

    def cross_entropy(self, logits, soft_targets):
        return th.mean(th.sum(- soft_targets * self.logsoftmax(logits), 1))