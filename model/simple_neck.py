import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hrformer.modules.bottleneck_block import Bottleneck, BottleneckDWP
from .hrformer.modules.transformer_block import GeneralTransformerBlock

class SimpleNeck(nn.Module):
    
    def __init__(self, pre_stage_channels):
        super().__init__()

        self.neck_size = sum(pre_stage_channels)
        self.maxpool = torch.nn.MaxPool2d(2)

    def pool(self, x):
        pad = [x.shape[2] % 2, x.shape[3] % 2]
        x = self.maxpool((
            F.pad(input=x, pad=(pad[1], 0     , pad[0], 0     ), mode='constant', value=0) +
            F.pad(input=x, pad=(pad[1], 0     , 0     , pad[0]), mode='constant', value=0) +
            F.pad(input=x, pad=(0     , pad[1], pad[0], 0     ), mode='constant', value=0) +
            F.pad(input=x, pad=(0     , pad[1], 0     , pad[0]), mode='constant', value=0)
        )/4)
        return x

    def forward(self, x):
        y = x[0]
        for i in range(1,len(x)):
            y = torch.cat((self.pool(y),x[i]),dim=1)
        return y