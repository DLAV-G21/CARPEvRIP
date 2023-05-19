import os
import torch
import torch.nn as nn

from .hrformer.hrt import HighResolutionTransformer
from .head import Head
from .neck import Neck
from .simple_neck import SimpleNeck
from .decoder import Decoder
import logging
class Net(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.log = logging.getLogger("g21")
        self.name = config['name']
        self.epoch = 0
        self.best_result = -1
        self.backbone = self.Load_Backbones()
        self.neck = self.Load_Neck(config, self.backbone.pre_stage_channels)
        self.keypoints = self.Load_Head(config, 2, config['dataset']['nb_keypoints'], self.neck.embed_size)
        self.links = self.Load_Head(config, 4, config['dataset']['nb_links'], self.neck.embed_size)
        if(config['model']['decode_output']):
            self.decoder = self.Load_Decoder(config)
        else:
            self.decoder = None
        self.init_weights(config['model']['backbone_save'], config['model']['model_saves'])
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

    def Load_Neck(self, config, pre_stage_channels):
        if(config['model']['simple_neck']):
            return SimpleNeck(pre_stage_channels, 
                       nhead = config['model']['nhead'],
                       num_layers = config['model']['num_layers'],
                       )
        return Neck(pre_stage_channels)

    def Load_Head(self, config, nbr_variable, nbr_points, embed_size):
        return Head(
            nbr_max_car=config['dataset']['max_nb'],
            nbr_points=nbr_points,
            nbr_variable=nbr_variable,
            nhead = config['model']['nhead'],
            num_layers = config['model']['num_layers'],
            use_matcher = config['model']['use_matcher'],
            normalize_position = config["dataset"]["normalize_position"],
            embed_size = embed_size,
        )

    def Load_Decoder(self, config):
        return Decoder(
            config['decoder']['threshold'],
            config['decoder']['max_distance'],
            use_matcher = config['model']['use_matcher']
        )

    def forward(self, x):
        if(self.train_backbone):
            x = self.backbone(x)
        else:
            with torch.no_grad():
                self.backbone.eval()
                x = self.backbone(x)
        x = self.neck(x)
        y = self.keypoints(x)
        z = self.links(x)
        if(self.decoder is not None and not self.training):
            return self.decoder((y, z))
        return y, z
    
    def init_weights(
        self,
        backbone_save,
        pretrained="",
    ):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if len(pretrained) > 0:
            pretrained = os.path.join(pretrained, self.name)
            if os.path.isdir(pretrained):
                files = [int(f[6:-4]) for f in os.listdir(pretrained) if (
                    f.startswith('model_') and 
                    f.endswith('.pth') and 
                    os.path.isfile(os.path.join(pretrained, f))
                    )]
                
                if(len(files) > 0):
                    pretrained = os.path.join(pretrained, f'model_{max(files)}.pth')
                else:
                    pretrained = False
            else:
                pretrained = False

        if os.path.isfile(pretrained):
            self.log.info("Loading pretrained parameters from "+pretrained)

            pretrained_dict = torch.load(pretrained, map_location='cpu')
            
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() if k in model_dict.keys()
            }

            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

            self.epoch = int(pretrained.split('model_')[-1][:-4])
        else:
            self.log.info("Initialised the weights")
            if(pretrained is not False) and len(pretrained) > 0:
                raise ValueError('The given pretrained model file doesn\'t exist :', pretrained)
            if not os.path.isfile(backbone_save):
                raise ValueError('The given backbone_save file doesn\'t exist :', backbone_save)
            self.backbone.init_weights(backbone_save)

    def save_weights(self, filename):
        torch.save(self.state_dict(), filename)