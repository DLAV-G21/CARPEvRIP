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
from pycocotools.coco import COCO
from PIL import Image
from functools import partial


class ApolloInference(Dataset):
  def __init__(self,root_path, img_path,config):
      self.coco = None
      self.img_path = os.path.join(root_path,img_path)
         # Get the mean and std of the dataset
      self.mean = config['data_augmentation']['normalize']['mean']
      self.std = config['data_augmentation']['normalize']['std']

      # Get the image size
      self.image_size = tuple(config['dataset']['input_size'])
      self.img_size = tuple(config['dataset']['img_size'])
      self.dataset = self.load_data()
    
  def __len__(self):
    return len(self.dataset)

  def load_data(self):
      dataset = []
      for f in os.listdir(self.img_path):
        if os.path.isfile(os.path.join(self.img_path, f)):
          dataset.append(f)
      return dataset
  
  def __getitem__(self, idx):
    img_name = self.dataset[idx]
    img = np.array(Image.open(os.path.join(self.img_path, img_name)))
    list_transform = [al.augmentations.geometric.resize.Resize(height=self.image_size[1], width=self.image_size[0],interpolation=cv2.INTER_CUBIC,always_apply=True, p=1.0)]
    list_transform.append(al.Normalize(mean=self.mean, std=self.std))
    list_transform.append(ToTensorV2())

    composition = al.Compose(list_transform)
    transformed = composition(image=img)
    transformed_image = transformed['image']
    return transformed_image, img_name
    


class ApolloEvalDataset(Dataset):
  def __init__(self, data_list, config, root_path, is_val,is_inference=False, inference_path="", inference_image_path=""):
    self.inference_path = inference_path
    self.is_inference = is_inference
    # Define paths for the different folders
    self.img_path = os.path.join(root_path,config['dataset']['img_path']) if not is_inference else os.path.join(root_path,inference_image_path)
    self.img_size = tuple(config['dataset']['img_size'])

    # Get the mean and std of the dataset
    self.mean = config['data_augmentation']['normalize']['mean']
    self.std = config['data_augmentation']['normalize']['std']

    # Get the image size
    self.image_size = tuple(config['dataset']['input_size'])

    # Get the width and height ratio
    self.width_ratio = self.image_size[0]/self.img_size[0]
    self.height_ratio = self.image_size[1]/self.img_size[1]
    
    self.normalize_position = config["dataset"]["normalize_position"]

    self.max_car = config["dataset"]["max_nb"]

    # Load the data
    self.dataset, self.annotation_file = self.load_data(root_path,config, data_list)

    # Create a coco file
    coco_path = os.path.join(root_path,"coco_val.json" if is_val else "coco_test.json") if not is_inference else os.path.join(root_path, inference_path)

    # Write the coco file
    with open(coco_path, 'w') as f:
      json.dump(self.annotation_file,f)
    
    # Create the COCO object
    self.coco = COCO(coco_path)
  
  def __len__(self):
    return len(self.dataset)

  def load_data(self, root_path,config, data_list,inference_path=""):
    root_path = os.path.join(root_path, config["dataset"]["annotations_folder"])
    
    train_file = 'apollo_keypoints_'+str(config['dataset']['nb_keypoints'])+'_'+'train'+'.json'
    val_file = 'apollo_keypoints_'+str(config['dataset']['nb_keypoints'])+'_'+'val'+'.json'
     # Check if given file exists
    if os.path.exists(os.path.join(root_path,train_file)):
      with open(os.path.join(root_path,train_file), 'r') as f:
        data_file = json.load(f)
    else:
      raise ValueError('The given config file doesn\'t exist :', os.path.join(root_path,train_file))
    # Check if given file exists
    if os.path.exists(os.path.join(root_path,val_file)):
      with open(os.path.join(root_path,val_file), 'r') as f:
        data_file2 = json.load(f)
    else:
      raise ValueError('The given config file doesn\'t exist :', os.path.join(root_path,val_file))
    

    if self.is_inference:
      data_file2 = {"images":[], "annotations":[]}
      if os.path.exists(os.path.join(root_path,inference_path)):
        with open(os.path.join(root_path,inference_path), 'r') as f:
          data_file = json.load(f)
      else:
        raise ValueError('The given config file doesn\'t exist :', os.path.join(root_path,inference_path))

    all_images = []

    all_images.extend(data_file["images"])
    all_images.extend(data_file2["images"])

    all_annotations = []
    all_annotations.extend(data_file["annotations"])
    all_annotations.extend(data_file2["annotations"])

    # Initialize empty data set
    dataset = []
    coco_annotations = {}
    coco_annotations["images"] = []
    coco_annotations["annotations"] = []
    # Copy categories and info
    coco_annotations["categories"] = data_file["categories"].copy()
    coco_annotations["info"] =  data_file["info"].copy()

    # Iterate over images
    for dico in all_images:
      im_id = dico["id"]
      # Check if image is in the given data list
      if dico["file_name"] in data_list:
        # Add image to coco annotations
        coco_annotations["images"].append(dico)
        # Initialize empty annotations list
        cur_annotations = []
        # Iterate over annotations
        for dico_2 in all_annotations:
          # Make copy of annotation
          dico_copy = dico_2.copy()
          # Check if annotation is for the current image and is not a crowd annotation
          if dico_2['image_id'] == im_id and dico_2["iscrowd"]==0:
            # Iterate over keypoints
            keypoints = []
            for i in range(24):
              # Get x, y and z values of keypoints
              x,y,z = tuple(dico_2['keypoints'][i*3:(i+1)*3])
              # Multiply by width and height ratio
              if self.normalize_position:
                x /= self.img_size[0]
                y /= self.img_size[1] 
              else:
                x *= self.width_ratio
                y *= self.height_ratio
              # Append to list of keypoints
              keypoints.extend([x,y,z])

            # Multiply box by width and height ratio
            # box is in   (x, y, w, h)
            dico_copy["bbox"] = [dico_copy["bbox"][0]/self.image_size[0] , dico_copy["bbox"][1]/self.image_size[1],dico_copy["bbox"][2]/self.image_size[0], dico_copy["bbox"][3]/self.image_size[1] ]
            dico_copy["scale"] = dico_copy['area']/(self.image_size[0]*self.image_size[1])
            # Assign new keypoints
            dico_copy["keypoints"] = keypoints
            # Append to current annotations
            cur_annotations.append(dico_copy)
          
        cur_annotations = sorted(cur_annotations, key=lambda x:x["scale"],reverse=True)[:self.max_car]
              
        # Append annotations and image to dataset
        dataset.append((cur_annotations, dico.copy()))  
        # Extend annotations of coco annotations
        coco_annotations["annotations"].extend(cur_annotations)

    return dataset, coco_annotations
  
  def __getitem__(self, idx):
    img_name = self.dataset[idx][1]["file_name"]
    img = np.array(Image.open(os.path.join(self.img_path, img_name)))
    list_transform = [al.augmentations.geometric.resize.Resize(height=self.image_size[1], width=self.image_size[0],interpolation=cv2.INTER_CUBIC,always_apply=True, p=1.0)]
    list_transform.append(al.Normalize(mean=self.mean, std=self.std))
    list_transform.append(ToTensorV2())

    composition = al.Compose(list_transform)
    transformed = composition(image=img)
    transformed_image = transformed['image']
    
    return transformed_image, self.dataset[idx][1]["id"]

class ApolloDataset(Dataset):
  def __init__(self, data_list, config, root_path):
    
    self.config = config
    seed = config['dataset']['seed']
    random.seed(seed)
    self.use_matcher = config["model"]["use_matcher"]
    self.normalize_position = config["dataset"]["normalize_position"]
    self.SCALE_SINGLE_KP = 1e-9
    self.max_nb_car = config['dataset']['max_nb']
    self.image_size = tuple(config['dataset']['input_size'])
    self.dataset = self.load_data(root_path, config, data_list)
    self.img_path = os.path.join(root_path,config['dataset']['img_path'])
    self.segm_path = os.path.join(root_path,config['dataset']['segm_path'])
    self.img_size = config['dataset']['img_size']
    self.use_occlusion_data_augm = config['data_augmentation']['use_occlusion_data_augm']
    self.apply_augm = config['data_augmentation']['apply_augm']
    self.mean = config['data_augmentation']['normalize']['mean']
    self.std = config['data_augmentation']['normalize']['std']
    self.prob_mask = config['data_augmentation']['prob_occlusion']
    self.prob_blur = config['data_augmentation']['prob_blur']
    self.nb_blur_source = config['data_augmentation']['nb_blur_source']
    self.blur_radius = config['data_augmentation']['blur_radius']
    
    
    self.nb_links = len(CAR_SKELETON_24) if config['dataset']['nb_keypoints'] == 24 else len(CAR_SKELETON_66)
    self.nb_keypoints = config['dataset']['nb_keypoints']
    self.list_links = CAR_SKELETON_24 if config['dataset']['nb_keypoints'] == 24 else CAR_SKELETON_66
    
  def load_data(self,root_path, config,data_list):
    root_path = os.path.join(root_path, config["dataset"]["annotations_folder"])
    train_file = 'apollo_keypoints_'+str(config['dataset']['nb_keypoints'])+'_'+'train'+'.json'
    val_file = 'apollo_keypoints_'+str(config['dataset']['nb_keypoints'])+'_'+'val'+'.json'
    print(os.path.join(root_path,train_file))
    if os.path.exists(os.path.join(root_path,train_file)):
      with open(os.path.join(root_path,train_file), 'r') as f:
        data_file = json.load(f)
    else:
      raise ValueError('The given config file doesn\'t exist :', os.path.join(root_path,train_file))
    if os.path.exists(os.path.join(root_path,val_file)):
      with open(os.path.join(root_path,val_file), 'r') as f:
        data_file2 = json.load(f)
    else:
      raise ValueError('The given config file doesn\'t exist :', os.path.join(root_path,val_file))

    dataset = []
    annotations = {}
    names = {}

    all_images = []
    all_images.extend(data_file["images"])
    all_images.extend(data_file2["images"])

    all_annotations = []
    all_annotations.extend(data_file["annotations"])
    all_annotations.extend(data_file2["annotations"])

    id_list = []
    for dico in all_images:
      im_id = dico["id"]
      if dico["file_name"] in data_list:
        id_list.append(im_id)
        names[im_id] = dico["file_name"]

    for dico in all_annotations:
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
            scales.append(max(self.SCALE_SINGLE_KP, ls['area']/(self.image_size[0]*self.image_size[1])))

      # Keep a fix number of car 
      kept = [(idx, scale) for idx, scale in enumerate(scales)]
      kept = sorted(kept,key=lambda x: x[1], reverse=True)[:self.max_nb_car]
      indices = [i for i, _ in kept]
      scales = [x for i, x in enumerate(scales) if i in indices]
      kps =[kp for i, kp in enumerate(kps)if i in indices]
      nb=min(len(kps), self.max_nb_car)
      name = names[im_id]
      if nb > 0:
        dataset.append((name, kps, scales, nb))

    return dataset

  def __len__(self):
    return len(self.dataset)

  def get_source_in_mask(self, mask):
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
    if not self.use_matcher:
      lst[:,:,0]=0

    for i in range(nb_car):
      label_car =[int(j.split('-')[1]) for j in labels if int(j.split('-')[0])==i]

      for cls, (a,b) in enumerate(lst_links):
        if a in label_car and b in label_car:
          idx1 = label_car.index(a)
          idx2 = label_car.index(b)
          pt10, pt11 = kps[idx1]
          pt20, pt21 = kps[idx2]
          if self.use_matcher:
            lst[i,cls,:] = torch.Tensor([cls+1, pt10,pt11, pt20, pt21])  
          else:
            lst[i,cls,:] = torch.Tensor([1, pt10, pt11, pt20, pt21])
    return lst 

  def keypoints_list_to_tensor(self, kps, labels, nb_car):
    keypoints = torch.ones((self.max_nb_car, self.nb_keypoints, 3))*(-1)
    if not self.use_matcher:
      keypoints[:,:,0]=0
      
    for i in range(nb_car):
      for idx, label in enumerate(labels):
        lab = label.split('-')
        if int(lab[0]) == i:
          kp = kps[idx]
          x, y = kp[0],kp[1]
          if self.use_matcher:
            keypoints[i,int(lab[1])-1,:]=torch.Tensor([int(lab[1]),x,y])
          else:
             keypoints[i,int(lab[1])-1,:]=torch.Tensor([1,x,y])
    return keypoints 

  def __getitem__(self, index):
    cur_name, keypoints, scales, nb_car = self.dataset[index]  
    scale = torch.Tensor(scales+[-1]*(self.max_nb_car-nb_car))
    img = np.array(Image.open(os.path.join(self.img_path,cur_name)))


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
          segm = np.load(os.path.join(self.segm_path, cur_name+".npz"))["arr_0"]
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
        al.RandomBrightnessContrast(p=0.5),
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

    if self.normalize_position: 
      transformed_keypoints = [(x/self.image_size[0],y/self.image_size[1]) for x, y in transformed_keypoints]

    links = self.find_link_among_keypoints(transformed_keypoints, transformed_class_labels, nb_car)
    keypoints = self.keypoints_list_to_tensor(transformed_keypoints, transformed_class_labels, nb_car)

    return cur_name, transformed_image, keypoints, scale, links, torch.Tensor([nb_car]).int()

def collate_fn(data, ann =True):
  if ann:
    img, ids, annots = [d[0] for d in data],[d[1] for d in data],[d[2] for d in data]
    return torch.utils.data.default_collate(img), ids, annots
  else:
    img, ids = [d[0] for d in data],[d[1] for d in data]
    return torch.utils.data.default_collate(img), ids

def get_dataloaders(config, data_path):
  train_data_list, val_data_list, test_data_list = generate_train_val_test_split(config, data_path)

  train_dataset = ApolloDataset(train_data_list, config, data_path)
  val_dataset =  ApolloEvalDataset(val_data_list, config, data_path, is_val =True)
  test_dataset =  ApolloEvalDataset(test_data_list, config, data_path, is_val =False)

  train_loader = DataLoader(train_dataset,batch_size=config['training']['batch_size'], num_workers=config['hardware']['num_workers'],shuffle=config['dataset']['shuffle'])
  val_loader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=config['training']['batch_size'], num_workers=config['hardware']['num_workers'],shuffle=False)
  test_loader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=config['training']['batch_size'],num_workers=config['hardware']['num_workers'],shuffle=False)
  
  return train_loader, val_loader, test_loader


def load_dataloader_inference(config, root_path, image_path, annotations_file=None):
    if annotations_file is None:
      dataset = ApolloInference(root_path, image_path,config)
      ann =False 
    else:
      data_list = [ f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]
      dataset =  ApolloEvalDataset(data_list, config, root_path, is_val=False, is_inference=True, inference_path=annotations_file, inference_image_path=image_path)
      ann = True
    
    return DataLoader(dataset, collate_fn=partial(collate_fn, ann=ann), batch_size=config['training']['batch_size'],num_workers=config['hardware']['num_workers'],shuffle=False)