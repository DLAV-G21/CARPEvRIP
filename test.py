import torch
from model.losses.loss_keypoints import LossKeypoints

def main():
    l = LossKeypoints()
    bs = 3
    predicted_keypoints = torch.rand((bs,240,27))
    targeted_keypoints = torch.rand((bs,10,24,3))
    scale = torch.rand((bs,10))
    nb_cars = torch.rand((bs,1))

    l(predicted_keypoints, targeted_keypoints, scale, nb_cars)

if __name__ == '__main__' :
    main()  