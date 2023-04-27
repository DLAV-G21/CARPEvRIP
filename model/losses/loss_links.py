import torch
import torch.nn as nn
from .matching import HungarianMatcher

MAX_DISTANCE = 100

class LossLinks(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
        """
        super().__init__()
        
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox

        self.matcher = HungarianMatcher(
            self.get_position_from_output,
            self.get_class_distribution_from_output,
            self.get_position_from_target,
            self.get_class_from_target,
            cost_class, cost_bbox)
        self.criterion = nn.CrossEntropyLoss()
        assert cost_class != 0 or cost_bbox != 0, "all costs cant be 0"

    def get_class_distribution_from_output(self, keypoint):
        return keypoint[:,:,4:]
    
    def get_class_from_output(self, keypoint):
        return torch.argmax(self.get_class_distribution_from_output(keypoint), dim=2)
    
    def get_position_from_output(self, keypoint):
        return keypoint[:,:,:4]
    
    def get_position_from_target(self, keypoint):
        return keypoint[:,:,:,1:].flatten(1, 2)
    
    def get_class_from_target(self, keypoint):
        return keypoint[:,:,:,0].flatten(1, 2)
    
    def compute_distance(self, predicted_keypoints, targeted_keypoints):
        return torch.sqrt(
                torch.sum((
                    self.get_position_from_output(predicted_keypoints) - 
                    self.get_position_from_target(targeted_keypoints)
                ) ** 2, dim=2)
            )
    
    def forward(self, predicted_keypoints, targeted_keypoints, scale, nb_cars):
        indices = self.matcher(predicted_keypoints, targeted_keypoints)

        bs = targeted_keypoints.shape[0]
        num_targets = targeted_keypoints.shape[1] * targeted_keypoints.shape[2]
        predicted_keypoints = predicted_keypoints.flatten(0,1)[indices[1].flatten(0,1),:].view(bs,num_targets,-1)

        distance = self.compute_distance(predicted_keypoints, targeted_keypoints)
        
        filter = self.get_class_from_target(targeted_keypoints).view(targeted_keypoints.shape[:3]) > 0

        sum_ = torch.sum(filter, dim=2).unsqueeze(2)
        sum_[sum_ < 1] = 1

        OKS_loss = torch.sum(nb_cars) - torch.sum(
            torch.exp(- distance.view(targeted_keypoints.shape[:3]) / scale.unsqueeze(2)**2 ) * filter / sum_
        )

        target_class = self.get_class_from_target(targeted_keypoints)
        target_class[target_class < 0] = 0

        classification_loss = self.criterion(self.get_class_distribution_from_output(predicted_keypoints).permute(0,2,1), target_class.long())

        return self.cost_bbox * OKS_loss + self.cost_class * classification_loss