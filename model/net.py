import os
import torch
import torch.nn as nn

from .models.hrt import HighResolutionTransformer
from .models.points_transformer import PointsTransformer
from .models.neck import Neck
from .decoder import Decoder

class Net(nn.Module):
    
    def __init__(self, config):
        super().__init__()

        self.backbone = self.Load_Backbones()
        self.neck = self.Load_Neck(self.backbone.pre_stage_channels)
        self.keypoints = self.Load_Keypoints(config)
        self.links = self.Load_Links(config)
        self.links = self.Load_Links(config)
        if(config['model']['decode_output']):
            self.decoder = self.Load_Decoder(config)
        else:
            self.decoder = None
        self.init_weights(config['model']['pretrained'])
        self.train_backbone = config['training']['train_backbone']
        
    def Load_Backbones(self):
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
        return backbone

    def Load_Neck(self, pre_stage_channels):
        return Neck(pre_stage_channels)

    def Load_Keypoints(self, config):
        return PointsTransformer(
            nbr_max_car=config['dataset']['max_nb'],
            nbr_points=config['dataset']['nb_keypoints'],
            nbr_variable=2,
            bn_momentum = config['model']['bn_momentum'],
        )

    def Load_Links(self, config):
        return PointsTransformer(
            nbr_max_car=config['dataset']['max_nb'],
            nbr_points=config['dataset']['nb_links'],
            nbr_variable=4,
            bn_momentum = config['model']['bn_momentum'],
        )

    def Load_Decoder(self, config):
        return Decoder(
            config['decoder']['threshold'],
            config['decoder']['max_distance'],
        )

    def forward(self, x):
        if(self.train_backbone):
            x = self.backbone(x)
        else:
            self.backbone.init_weights("hrt_small_coco_384x288.pth")
            with torch.no_grad():
                x = self.backbone(x)
        x = self.neck(x)
        y = self.keypoints(x)
        z = self.links(x)
        if(self.decoder is not None and not self.training):
            return self.decoder((y, z))
        return (y, z)
    
    def init_weights(
        self,
        pretrained="",
    ):
        self.epoch = 0

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained, map_location='cpu')
            
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() if k in model_dict.keys()
            }

            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        else:
            self.backbone.init_weights("hrt_small_coco_384x288.pth")

    def save_weights(self, filename):
        torch.save(self.state_dict(), filename)