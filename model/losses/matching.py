# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
https://github.com/facebookresearch/detr/blob/main/models/matcher.py
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, 
                 get_position_from_output, get_class_distribution_from_output, get_position_from_target, get_class_from_target,
                 cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, max_distance = 100):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()

        self.get_position_from_output = get_position_from_output
        self.get_position_from_target = get_position_from_target

        self.get_class_distribution_from_output = get_class_distribution_from_output
        self.get_class_from_target = get_class_from_target

        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.max_distance = max_distance
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        indices = []
        for i in range(outputs.shape[0]):
            indices.append(self.forward_(outputs[i].unsqueeze(0), targets[i].unsqueeze(0)))
        return torch.cat(indices, dim=1)

    @torch.no_grad()
    def forward_(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs.shape[:2]
        num_targets = targets.shape[1] * targets.shape[2]

        # We flatten to compute the cost matrices in a batch
        out_prob = self.get_class_distribution_from_output(outputs).flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = self.get_position_from_output(outputs).flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = self.get_class_from_target(targets).flatten(0, 1).long()
        tgt_bbox = self.get_position_from_target(targets).flatten(0, 1)

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L2 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=2)
        cost_bbox[cost_bbox > self.max_distance] = self.max_distance
        cost_bbox[:, tgt_ids <= 0] = self.max_distance
        
        cost_bbox = cost_bbox.T
        cost_class = cost_class.T

        # Compute the giou cost betwen boxes
        # cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class# + self.cost_giou * cost_giou
        C = C.view(bs, num_targets, -1).cpu()

        sizes = [num_queries for _ in range(bs)]
        indices = torch.as_tensor(np.array([linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))])).permute(1, 0, 2)
        return indices #[(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]