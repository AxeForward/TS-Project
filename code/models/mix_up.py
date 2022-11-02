# FCN Module Structure

import torch.nn as nn

#@title Define FCN

class FCN(nn.Module):
    def __init__(self, n_in):
        super(FCN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(n_in, 128, kernel_size=7, padding=6, dilation=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=8, dilation=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=8, dilation=8),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        self.proj_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, x):

        h = self.encoder(x)
        out = self.proj_head(h)

        return out, h