import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models.hrt import HighResolutionTransformer
from .models.points_net import PointsTransformer
from .models.modules.bottleneck_block import Bottleneck, BottleneckDWP

BN_MOMENTUM = 0.1

class Net(nn.Module):
    
    def __init__(self, pretrained=True):
        super(Net, self).__init__()
        self.backbone =  self.Load_Backbone()
        self.keypoints = self.Load_Keypoints(pretrained, self.backbone.pre_stage_channels)
        self.links = self.Load_Links(pretrained, self.backbone.pre_stage_channels)
        
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

        for i in range(4):
            layer_config = cfg["STAGE" + str(1+i)]
            num_modules = layer_config["NUM_MODULES"]
            num_branches = layer_config["NUM_BRANCHES"]
            num_blocks = layer_config["NUM_BLOCKS"]

            layer_config["NUM_RESOLUTIONS"] = [None for branch_index in range(num_branches)]
            layer_config["ATTN_TYPES"] = [[["isa_local" for blocks_index in range(num_blocks[branch_index])] for branch_index in range(num_branches)] for module_index in range(num_modules)]
            layer_config["FFN_TYPES"] = [[["conv_mlp" for blocks_index in range(num_blocks[branch_index])] for branch_index in range(num_branches)] for module_index in range(num_modules)]

            cfg["STAGE" + str(1+i)] = layer_config


        backbone = HighResolutionTransformer(cfg)
        backbone.init_weights("hrt_small_coco_384x288.pth" if (pretrained) else "")
        return backbone

    def Load_Keypoints(self, pretrained, pre_stage_channels):
        return PointsTransformer(pre_stage_channels,
        nbr_points=24,
        nbr_variable=2
        )

    def Load_Links(self, pretrained, pre_stage_channels):
        head_block = BottleneckDWP
        head_channels = [32, 64, 128, 256]

        return lambda x: x

    def forward(self, x):
        x = self.backbone(x)
        y = self.keypoints(x)
        z = self.links(x)
        return [y, z]