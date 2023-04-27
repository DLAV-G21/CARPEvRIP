import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.bottleneck_block import Bottleneck, BottleneckDWP
from .modules.transformer_block import GeneralTransformerBlock

class PointsTransformer(nn.Module):
    def __init__(
        self,
        nbr_max_car,
        nbr_points,
        nbr_variable,
        bn_momentum = 0.1,
    ):
        super(PointsTransformer, self).__init__()

        self.nbr_max_car = nbr_max_car
        self.nbr_points = nbr_points
        self.nbr_variable = nbr_variable
        self.bn_momentum = bn_momentum


        head_size = 1024

        self.rescale = nn.Sequential(
            nn.Conv1d(
                in_channels=320,
                out_channels=nbr_points * nbr_max_car,
                kernel_size=1
            ),
            nn.BatchNorm1d(nbr_points * nbr_max_car, momentum=bn_momentum),
            nn.ReLU(inplace=True),
        )

        self.queries = torch.nn.Parameter(torch.rand(
            1,
            nbr_points * nbr_max_car,
            head_size,
        ))

        decoder_layer = nn.TransformerDecoderLayer(d_model=head_size, nhead=1)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

        self.final = nn.Conv1d(
            in_channels=head_size,
            out_channels=nbr_points + 1 + nbr_variable,
            kernel_size=1
        )
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        queries = self.queries.expand(x.shape[0], *self.queries.shape[1:3])
        x = x.view(*x.shape[0:2],-1)
        x = x.permute(0,2,1)
        x = self.rescale(x)

        x = self.transformer_decoder(queries, x)

        x = x.permute(0,2,1)
        x = self.final(x)
        x = x.permute(0,2,1)

        x[:,:,self.nbr_variable:] = \
            self.softmax(x[:,:,self.nbr_variable:])
        
        return x
        

        