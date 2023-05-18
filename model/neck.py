import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hrformer.modules.bottleneck_block import Bottleneck, BottleneckDWP
from .hrformer.modules.transformer_block import GeneralTransformerBlock

BN_MOMENTUM = 0.1

class Neck(nn.Module):
    def __init__(
        self,
        pre_stage_channels
    ):
        super().__init__()

        self.neck_size = 1024
        self.incre_modules, self.downsamp_modules = self._make_head(pre_stage_channels)
        
    def _make_head(self, pre_stage_channels):
        head_block = BottleneckDWP
        head_channels = pre_stage_channels
        
        # Increasing the #channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_layer(
                head_block, channels, head_channels[i], 1, stride=1
            )
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i + 1] * head_block.expansion
            downsamp_module = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=in_channels,
                ),
                nn.BatchNorm2d(in_channels, momentum=BN_MOMENTUM),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            )
            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        return incre_modules, downsamp_modules
    
    def _make_layer(
        self,
        block,
        inplanes,
        planes,
        blocks,
        num_heads=1,
        stride=1,
        window_size=7,
        halo_size=1,
        mlp_ratio=4.0,
        q_dilation=1,
        kv_dilation=1,
        sr_ratio=1,
        attn_type="msw",
    ):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = []

        if isinstance(block, GeneralTransformerBlock):
            layers.append(
                block(
                    inplanes,
                    planes,
                    num_heads,
                    window_size,
                    halo_size,
                    mlp_ratio,
                    q_dilation,
                    kv_dilation,
                    sr_ratio,
                    attn_type,
                )
            )
        else:
            layers.append(block(inplanes, planes, stride, downsample))

        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.incre_modules[0](x[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i + 1](x[i + 1]) + self.downsamp_modules[i](y)
        return y
        