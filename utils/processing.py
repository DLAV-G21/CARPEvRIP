import os
import random

import torchvision
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.patches as patches
from tqdm.notebook import tqdm



def generate_image_segmentation(img_path,save_path_img, save_path_segm,sample_demonstration=False):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
  model = model.to(device)
  model.eval()
  for x in tqdm(os.listdir(img_path)):
    img2 = np.array(Image.open(os.path.join(img_path,x)))
    img = torch.Tensor(img2).to(device).permute(2,0,1).unsqueeze(0)/255
    with torch.no_grad():
      predictions = model(img)
    
    mask = np.zeros((img2.shape[0],img2.shape[1]))
    if sample_demonstration:
      fig, ax = plt.subplots(1,2,figsize=(12,9))
      ax[0].imshow(img2)
    for idx, i in enumerate(predictions[0]["boxes"]):
      if int(predictions[0]["labels"][idx]) == 3 and predictions[0]["scores"][idx] > 0.95:
        
        i = [int(j) for j in list(i.detach().cpu().numpy())]
        mask[i[1]:i[3],i[0]:i[2]] = 1
        if sample_demonstration:
          rect = patches.Rectangle((i[0],i[1]), i[2]-i[0], i[3]-i[1], linewidth=5, edgecolor="r",facecolor="none")

          ax[0].add_patch(rect)
    if sample_demonstration:
      ax[1].imshow(mask*255,cmap="gray", interpolation="nearest")
      plt.show()
      return
    
    #np.save(os.path.join(save_path_img, x[:-4]+".npy"), img2)
    np.save(os.path.join(save_path_segm,x[:-4]+".npy"), mask)

    del(img2)
    del(img)
    del(predictions)

def generate_train_val_test_split(config,root_path):
  """
  The given validation dataset will give our testing set. 
  Then, we will use part of the train_split to make our validation dataset.
  """
  train_path =os.path.join(root_path,config["dataset"]["annotations_folder"],"train-list.txt")
  test_path =os.path.join(root_path,config["dataset"]["annotations_folder"],"validation-list.txt")
  if os.path.exists(train_path) and os.path.exists(test_path):
    random.seed(config["dataset"]["seed"])
    with open(train_path, "r") as ff:
      lines = ff.readlines()
    all_lines = [line.strip() for line in lines]
    nb_train = int(len(all_lines)*config["dataset"]["train_ratio"])
    random.shuffle(all_lines)
    train_data_list = all_lines[:nb_train]
    val_data_list = all_lines[nb_train:]

    with open(test_path, "r") as ff:
      lines = ff.readlines()
    test_data_list = [line.strip() for line in lines]

    return train_data_list, val_data_list, test_data_list
  else:
    raise ValueError("The given path for the training and testing files don't exist.")

  
