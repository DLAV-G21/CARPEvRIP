import json
from pickle import FALSE
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import os
import numpy as np

def plot_distribution(distribution,title,xlabel,ylabel="Count",logy=False,bins=None):
  if bins is None:
    bins = np.linspace(0, max(distribution),50)
  plt.hist(distribution, bins=bins,log=logy)

  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  plt.plot()

def plot_and_save_keypoints_inference(image_folder, data, output_folder, scale):
    for f in os.listdir(image_folder):
        if os.path.isfile(os.path.join(image_folder, f)):
            img = np.array(Image.open(os.path.join(image_folder, f)))
            plt.imshow(img)
            for lst in data[f]:
               kps = lst["keypoints"]

               for i in range(24):
                x,y,z = tuple(kps['keypoints'][i*3:(i+1)*3])
                x *= img.shape[1]/scale[0]
                y *= img.shape[2]/scale[1]
                if(z > 0):
                    circle1 = plt.Circle((x,y), 3, color='r')
                    plt.add_patch(circle1)

            plt.imsave(os.path.join(output_folder, f), img)
            plt.clf()

def plot_keypoints(im, keypoints):
  fig, axs = plt.subplots(1,2, figsize=(16,16))
  axs[0].imshow(im)
  axs[1].imshow(im)

  for kps in keypoints:
    for kp in kps:
      if kp[0]!=-1:
        x,y =kp[1],kp[2]
        circle1 = plt.Circle((x,y), 3, color='r')
        axs[1].add_patch(circle1)
  plt.show()

def plot_keypoints_and_bounding_box(image_id,config,root_path):
  annotations = {}
  lst = []
  image_path = [cnf["file_name"] for cnf in config["images"] if cnf["id"]==image_id][0]
  im = Image.open(os.path.join(root_path,"3d-car-understanding-train","train","images",image_path))
  
  for dico in config["annotations"]:
    
    im_id = dico["image_id"]
    #annotations[im_id]=annotations.get(im_id,[])+[dico.copy()]
    if dico["image_id"]==image_id:
      lst.append(dico)
  fig, axs = plt.subplots(1,3, figsize=(16,16))

  # Display the image
  axs[0].imshow(im)
  axs[1].imshow(im)
  axs[2].imshow(im)
  axs[0].axis('off')
  axs[1].axis('off')
  axs[2].axis('off')
  
  for ls in lst:
    if ls["iscrowd"] ==0:
      # Create a Rectangle patch
      bb = ls["bbox"]
      rect = patches.Rectangle((bb[0], bb[1]), bb[2], bb[3], linewidth=1, edgecolor='r', facecolor='none')

      # Add the patch to the Axes
      axs[1].add_patch(rect)
      for i in range(24):
        x,y,z = tuple(ls["keypoints"][i*3:(i+1)*3])
        circle1 = plt.Circle((x,y), 10, color='r')
        axs[2].add_patch(circle1) 