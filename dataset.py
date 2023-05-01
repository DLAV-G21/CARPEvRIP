import albumentations as al
from torch.utils.data import Dataset, DataLoader
import random
import torch
from albumentations.pytorch.transforms import ToTensorV2
import random
import cv2
import numpy as np
from utils.processing import generate_train_val_test_split
from utils.openpifpaf_helper import *
import os
import json

class ApolloDataset(Dataset):
  def __init__(self, data_list, config, root_path, is_training =True):
    
    self.config = config
    seed = config['dataset']['seed']
    random.seed(seed)
    self.SCALE_SINGLE_KP = 1e-8

    correct_file = 'apollo_keypoints_'+str(config['dataset']['nb_keypoints'])+'_'+('train' if is_training else 'val')+'.json'
    self.dataset = self.load_data(os.path.join(root_path, config['dataset']['annotations_folder'],correct_file), data_list)
    self.img_path = os.path.join(root_path,config['dataset']['img_path'])
    self.segm_path = os.path.join(root_path,config['dataset']['segm_path'])
    self.use_occlusion_data_augm = config['data_augmentation']['use_occlusion_data_augm'] and is_training
    self.apply_augm = config['data_augmentation']['apply_augm'] and is_training
    self.mean = config['data_augmentation']['normalize']['mean']
    self.std = config['data_augmentation']['normalize']['std']
    self.prob_mask = config['data_augmentation']['prob_occlusion']
    self.prob_blur = config['data_augmentation']['prob_blur']
    self.nb_blur_source = config['data_augmentation']['nb_blur_source']
    self.blur_radius = config['data_augmentation']['blur_radius']
    self.image_size = tuple(config['dataset']['input_size'])
    self.max_nb_car = config['dataset']['max_nb']
    self.nb_links = len(CAR_SKELETON_24) if config['dataset']['nb_keypoints'] == 24 else len(CAR_SKELETON_66)
    self.nb_keypoints = config['dataset']['nb_keypoints']
    self.list_links = CAR_SKELETON_24 if config['dataset']['nb_keypoints'] == 24 else CAR_SKELETON_66
    
  def load_data(self, file, data_list):
    if os.path.exists(file):
      with open(file, 'r') as f:
        data_file = json.load(f)
    else:
      raise ValueError('The given config file doesn\'t exist')

    dataset = []
    annotations = {}
    names = {}

    id_list = []
    for dico in data_file["images"]:
      im_id = dico["id"]
      if dico["file_name"] in data_list:
        id_list.append(im_id)
        names[im_id] = dico["file_name"][:-4]+".npy"

    for dico in data_file['annotations']:
      im_id = dico['image_id']
      if im_id in id_list:
        annotations[im_id]=annotations.get(im_id,[])+[dico.copy()]   
    

    for im_id, lst in annotations.items():
      kps = []
      scales = []

      for ls in lst:
        kps_car = []
        if ls['iscrowd'] == 0:
          for i in range(24):
            x,y,z = tuple(ls['keypoints'][i*3:(i+1)*3])
            if( z == 2.0):
              # the point is visible
              cls = i+1
              kps_car.append((cls,x,y))
          if len(kps_car)>0:
            kps.append(kps_car)
            if len(kps_car)==1:
              scales.append(self.SCALE_SINGLE_KP)
            else:
              scales.append(ls['area']/(3384*2710))
      nb=len(kps)
      name = names[im_id]
      if nb > 0:
        dataset.append((name, kps, scales, nb))
      

    return dataset

  def __len__(self):
    return len(self.dataset)

  def get_source_in_mask(self, mask):
    sources = []
    coordinates = np.transpose((mask==1).nonzero())
    samples = random.sample(np.arange(len(coordinates)), k=self.nb_blur_source)
    return coordinates[samples,:]

  def generate_image_segmentation(self, img,mask):
    blurred_img = cv2.GaussianBlur(img, (21, 21), 0)
    sources = self.get_source_in_mask(mask)
    mask_2 = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
    for s in sources:
      mask_2 = cv2.circle(mask_2, (s[0], s[1]), self.blur_radius, 255, -1)

    out = np.where(mask_2==255, img, blurred_img)
    return out
  
  def find_link_among_keypoints(self, kps,labels,nb_car):
    lst_links = self.list_links
    lst = torch.ones((self.max_nb_car, self.nb_links, 5))*(-1)
    for i in range(nb_car):
      label_car =[int(j.split('-')[1]) for j in labels if int(j.split('-')[0])==i]
      cpt = 0
      for cls, (a,b) in enumerate(lst_links):
        if a in label_car and b in label_car:
          idx1 = label_car.index(a)
          idx2 = label_car.index(b)
          pt10, pt11 = kps[idx1]
          pt20, pt21 = kps[idx2]
          lst[i,cpt,:] = torch.Tensor([cls, pt10, pt11, pt20, pt21])
          cpt+=1

    return lst 

  def keypoints_list_to_tensor(self, kps, labels, nb_car):
    keypoints = torch.ones((self.max_nb_car, self.nb_keypoints, 3))*(-1)
    for i in range(nb_car):
      cpt= 0
      for idx, label in enumerate(labels):
        lab = label.split('-')
        if int(lab[0]) == i:
          kp = kps[idx]
          keypoints[i,cpt,:]=torch.Tensor([int(lab[1]),kp[0],kp[1]])
          cpt +=1
    return keypoints 

  def __getitem__(self, index):
    cur_name, keypoints, scales, nb_car = self.dataset[index]  
    scale = torch.Tensor(scales+[-1]*(self.max_nb_car-nb_car))
    img = np.load(os.path.join(self.img_path, cur_name))["arr_0"]

    list_transform = [al.augmentations.geometric.resize.Resize(height=self.image_size[1], width=self.image_size[0],interpolation=cv2.INTER_CUBIC,always_apply=True, p=1.0)]
    if self.apply_augm:

      if self.use_occlusion_data_augm:
        if random.random() < self.prob_mask:
          RECT_SIZE = 16
          MAX_RECT = 11
          nb_mask = random.randint(MAX_RECT)
          fill_value = np.array([128, 128, 128])

          indices_hor = random.choices(list(range(0, self.input_size[0]//RECT_SIZE)),k=nb_mask)
          indices_ver = random.choices(list(range(0, self.input_size[1]//RECT_SIZE)),k=nb_mask)

          for i, j in zip(indices_hor, indices_ver):
            img[i*RECT_SIZE:(i+1)*RECT_SIZE, j*RECT_SIZE:(j+1)*RECT_SIZE] = fill_value
        
        if random.random() < self.prob_blur:
          segm = np.load(os.path.join(self.segm_path, cur_name))["arr_0"]
          if random.random() < 0.5:
            # Blur the cars
            segm = segm.astype(np.uint8)
          else:
            # blur the background
            segm = (segm==0).astype(np.uint8)
          img = self.generate_image_segmentation(img, segm)
      
      list_transform.extend([
        al.HorizontalFlip(p=0.5),
        al.ColorJitter(0.4, 0.4, 0.5, 0.2, p=0.6),
        al.ToGray(p=0.01),
        al.JpegCompression(50, 80,p=0.1),
        al.GaussNoise(var_limit=(1.0,30.0), p=0.2)
      ])
        
    list_transform.append(al.Normalize(mean=self.mean, std=self.std))
    list_transform.append(ToTensorV2())

    all_keypoints = []
    class_labels = []
    
    for id, kps in enumerate(keypoints):
      for cls, x, y in kps:
        class_labels.append(f'{id}-{cls}')
        all_keypoints.append((x,y))

    composition = al.Compose(list_transform, keypoint_params=al.KeypointParams(format='xy',label_fields=['class_labels']))
    transformed = composition(image=img, keypoints=all_keypoints, class_labels=class_labels)
    transformed_image = transformed['image']
    transformed_keypoints = transformed['keypoints']
    transformed_class_labels = transformed['class_labels']


    links = self.find_link_among_keypoints(transformed_keypoints, transformed_class_labels, nb_car)
    keypoints = self.keypoints_list_to_tensor(transformed_keypoints, transformed_class_labels, nb_car)

    return cur_name, transformed_image, keypoints, scale, links, torch.Tensor([nb_car]).int()

def get_dataloaders(config, data_path):
  train_data_list, val_data_list, test_data_list = generate_train_val_test_split(config, data_path)

  train_dataset = ApolloDataset(train_data_list, config, data_path, is_training =True)
  val_dataset =  ApolloDataset(val_data_list, config, data_path, is_training =True)
  test_dataset =  ApolloDataset(test_data_list, config, data_path, is_training =False)

  train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], num_workers=config['hardware']['num_workers'],shuffle=config['dataset']['shuffle'])
  val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], num_workers=config['hardware']['num_workers'],shuffle=config['dataset']['shuffle'])
  test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'],num_workers=config['hardware']['num_workers'],shuffle=config['dataset']['shuffle'])
  
  return train_loader, val_loader, test_loader