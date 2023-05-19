import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hrformer.modules.bottleneck_block import Bottleneck, BottleneckDWP
from .hrformer.modules.transformer_block import GeneralTransformerBlock

class SimpleNeck(nn.Module):
    
    def __init__(self, 
                 pre_stage_channels,
                 nhead = 4,
                 num_layers = 6,
                 ):
        super().__init__()

        embed_size = 512
        self.cat_size = sum(pre_stage_channels)
        self.embed_size = embed_size
        self.maxpool = nn.MaxPool2d(2)

        self.transformer_ecoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_size, nhead=nhead, batch_first=True),
            num_layers=num_layers)

    def pool(self, x):
        pad = [x.shape[2] % 2, x.shape[3] % 2]
        x = self.maxpool((
            F.pad(input=x, pad=(pad[1], 0     , pad[0], 0     ), mode='constant', value=0) +
            F.pad(input=x, pad=(pad[1], 0     , 0     , pad[0]), mode='constant', value=0) +
            F.pad(input=x, pad=(0     , pad[1], pad[0], 0     ), mode='constant', value=0) +
            F.pad(input=x, pad=(0     , pad[1], 0     , pad[0]), mode='constant', value=0)
        )/4)
        return x
    
    def cat_positional_encoding(self, x):
        #Gets the device and the data type of x
        device = x.device
        dtype = x.dtype
        list_ = [x]

        for i in range(1,1+(self.embed_size - self.cat_size)//2):
            #Creates a tensor with the range of x's shape[2]
            x_  = torch.tensor([(j//i) for j in range(x.shape[2])], dtype=dtype, device=device).expand(x.shape[0],1,x.shape[3],x.shape[2]).permute(0,1,3,2)
            #Creates a tensor with the range of x's shape[3]
            y_  = torch.tensor([(j//i) for j in range(x.shape[3])], dtype=dtype, device=device).expand(x.shape[0],1,x.shape[2],x.shape[3])
            list_.append(x_)
            list_.append(y_)

        x = torch.cat(list_,dim=1)
        #Reshapes x
        return x.view(*x.shape[0:2],-1)

    def forward(self, x):
        y = x[0]
        for i in range(1,len(x)):
            y = torch.cat((self.pool(y),x[i]),dim=1)
        y = self.cat_positional_encoding(y)
        y = y.permute(0,2,1)
        y = self.transformer_ecoder(y)
        return y