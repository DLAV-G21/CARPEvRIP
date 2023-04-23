import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models.hrt import HighResolutionTransformer

class net(nn.Module):
    
    def __init__(self, pretrained=True):
        self.Load_Backbone(pretrained)
        
    def Load_Backbone(self, pretrained=True):
        cfg = dict(
            drop_path_rate=0.1,
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(2, ),
                num_channels=(64, ),
                num_heads=[2],
                num_mlp_ratios=[4]),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='TRANSFORMER_BLOCK',
                num_blocks=(2, 2),
                num_channels=(32, 64),
                num_heads = [1, 2],
                num_mlp_ratios = [4, 4],
                num_window_sizes = [7, 7]),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='TRANSFORMER_BLOCK',
                num_blocks=(2, 2, 2),
                num_channels=(32, 64, 128),
                num_heads = [1, 2, 4],
                num_mlp_ratios = [4, 4, 4],
                num_window_sizes = [7, 7, 7]),
            stage4=dict(
                num_modules=2,
                num_branches=4,
                block='TRANSFORMER_BLOCK',
                num_blocks=(2, 2, 2, 2),
                num_channels=(32, 64, 128, 256),
                num_heads = [1, 2, 4, 8],
                num_mlp_ratios = [4, 4, 4, 4],
                num_window_sizes = [7, 7, 7, 7])
        )

        def toUpper(dict_):
            dict_upper = {}
            for k in dict_:
                if type(dict_[k]) is dict:
                    dict_upper[k.upper() if type(k) is str else k] = toUpper(dict_[k])
                else:
                    dict_upper[k.upper() if type(k) is str else k] = dict_[k]
            return dict_upper

        cfg = toUpper(cfg)

        self.backbone = HighResolutionTransformer(cfg)
        self.backbone.init_weights("hrt_small_coco_384x288.pth" if (pretrained) else "")

    def forward(self, x):
        x = self.backbone(x)