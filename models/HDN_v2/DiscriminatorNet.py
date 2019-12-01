import torch
import torch.nn as nn
import pdb
import json
import os
import os.path as osp
import re
import numpy as np

class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """

    def __init__(self, opts=None, args=None):
        super(DiscriminatorNet, self).__init__()
        n_features = 3*64*64
        n_out = 1

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(256, n_out),
            torch.nn.Sigmoid()
        )

        self.n_features = n_features

    def forward(self, x):
        if x.size(-1) != 64:
            x = nn.Upsample(size=[64,64], mode='nearest')(x)

        x = x.view(x.size(0), -1)
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
