import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hrt import HighResolutionTransformer
from .points_transformer import PointsTransformer
from .neck import Neck

class Decoder(nn.Module):
    
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.threshold = 0.5

        
    def get_class_distribution_from_keypoints(self, keypoint):
        return self.softmax(keypoint[:,:,2:])
    
    def get_class_distribution_from_links(self, keypoint):
        return self.softmax(keypoint[:,:,4:])
    
    def get_class_from_distribution(self, distribution):
        return torch.topk(distribution, k=1, dim=2)

    def forward(self, x):
        if self.training:
            return x
        
        keypoints, links = x

        res = []
        for b in range(keypoints.shape[0]):
            res_b = []
            keys = {}
            links = {}
            
            x[:,:,self.nbr_variable:] = \
                self.softmax(x[:,:,self.nbr_variable:])
            for key in range(y.shape[1]):
                

            res.append(res_b)
        return res

