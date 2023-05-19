import torch
import torch.nn as nn
from .matching import HungarianMatcher

class LossKeypoints(nn.Module):
    def __init__(self, nbr_variable, scale_factor = 1, cost_class: float = 1, cost_distance: float = 1, cost_OKS: float = 1, max_distance = 100, use_matcher = True):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
        """
        super().__init__()
        self.nbr_variable = nbr_variable
        self.cost_class = cost_class
        self.cost_distance = cost_distance
        self.cost_OKS = cost_OKS
        self.use_matcher = use_matcher
        self.scale_factor = scale_factor

        self.matcher = HungarianMatcher(
            self.get_position_from_output,
            self.get_class_distribution_from_output,
            self.get_position_from_target,
            self.get_class_from_target,
            cost_class, cost_distance+cost_OKS, max_distance)  if use_matcher else None
        self.criterion = nn.CrossEntropyLoss() if use_matcher else nn.BCEWithLogitsLoss()
        assert cost_class != 0 or (cost_distance+cost_OKS) != 0, "all costs cant be 0"

    def get_class_distribution_from_output(self, keypoint):
        return keypoint[:,:,self.nbr_variable:] if self.use_matcher else keypoint[:,:,self.nbr_variable]
    
    def get_position_from_output(self, keypoint):
        return keypoint[:,:,:self.nbr_variable]
    
    def get_position_from_target(self, keypoint):
        return keypoint[:,:,:,1:].flatten(1, 2)
    
    def get_class_from_target(self, keypoint):
        return keypoint[:,:,:,0].flatten(1, 2)
    
    def compute_distance(self, predicted_keypoints, targeted_keypoints):
        #TODO ou pas
        return  torch.sum(torch.abs(
                    self.get_position_from_output(predicted_keypoints) - 
                    self.get_position_from_target(targeted_keypoints)
                ), dim=2)
    
    def forward(self, predicted_keypoints, targeted_keypoints, scale, nb_cars):
        bs = targeted_keypoints.shape[0]
        num_targets = targeted_keypoints.shape[1] * targeted_keypoints.shape[2]
        
        if(self.use_matcher):
            indices = self.matcher(predicted_keypoints, targeted_keypoints)
            predicted_keypoints = predicted_keypoints.flatten(0,1)[indices[1].flatten(0,1),:].view(bs,num_targets,-1)

        distance = self.compute_distance(predicted_keypoints, targeted_keypoints).view(targeted_keypoints.shape[:3])
        
        filter_ = (self.get_class_from_target(targeted_keypoints).view(targeted_keypoints.shape[:3]) > 0).int()

        sum_ = torch.sum(filter_, dim=2).unsqueeze(2)
        sum_[sum_ < 1] = 1
        # Mean over all
        distance_loss = torch.sum(distance * filter_ / sum_ / nb_cars.expand(bs,num_targets).view(filter_.shape))

        OKS_loss = 0
        
        if self.cost_OKS > 0:
            OKS_loss = torch.sum(nb_cars) - torch.sum(
                torch.exp(- distance / (scale.unsqueeze(2)**2 * self.scale_factor) ) * filter / sum_
            )

        target_class = self.get_class_from_target(targeted_keypoints)
        target_class[target_class < 0] = 0

        classification_loss = self.criterion(
            self.get_class_distribution_from_output(predicted_keypoints).permute(0,2,1) if self.use_matcher else self.get_class_distribution_from_output(predicted_keypoints),
            target_class.long() if self.use_matcher else target_class
        )

        return  self.cost_distance * distance_loss + self.cost_OKS * OKS_loss + self.cost_class * classification_loss