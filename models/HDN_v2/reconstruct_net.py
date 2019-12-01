import torch
import torch.nn as nn
import pdb
import json
import os
import os.path as osp
import re
import numpy as np

import torch.nn.functional as F
from .CRN import RefinementNetwork


class RC_net(nn.Module):
    def __init__(self, opts=None, args=None):
        super(RC_net, self).__init__()
        self.opts = opts
        self.args = args

        self.expand = nn.Linear(1, 64)
        self.up = nn.Upsample(scale_factor=8, mode='nearest')
        self.crn = RefinementNetwork(dims=[516, 256, 128, 32], opts=opts, args=args)

    def forward(self, pooled_object_features, object_rois):
        obj_expand = torch.cat([pooled_object_features, object_rois[:,1:]], -1)
        obj_expand = self.expand(obj_expand.unsqueeze(-1))
        obj_expand = obj_expand.view(
            obj_expand.size(0), obj_expand.size(1), 8, 8)
        obj_expand = self.up(obj_expand)
        obj_expand = torch.sum(obj_expand, 0, keepdim = True)

        out = self.crn(obj_expand)
        
        return out
