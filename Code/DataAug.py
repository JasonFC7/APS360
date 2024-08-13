from PIL import Image
from IPython.display import display
import os
from os import listdir
import torch
from torchvision.transforms import v2

# Enter original and save folder
folder_dir = "/content/cdr3"
save_folder = "/content/cdr3_aug"

# Choose a transformation
transforms1 = v2.Compose([v2.RandomVerticalFlip(p=1),])
transforms2 = v2.Compose([v2.RandomHorizontalFlip(p=1),])
transforms3 = v2.Compose([v2.RandomRotation(30)])
transforms4 = v2.Compose([v2.RandomRotation(35)])
transforms5 = v2.Compose([v2.RandomRotation(40)])
transforms6 = v2.Compose([v2.RandomRotation(45)])
transforms7 = v2.Compose([v2.RandomRotation(50)])
transforms8 = v2.Compose([v2.RandomRotation(55)])
transforms9 = v2.Compose([v2.RandomRotation(60)])

for image in os.listdir(folder_dir):
  image_path = os.path.join(folder_dir, image) # Construct the full image path
  visual = Image.open(image_path)
  flip = transforms1(visual)
  save_path = os.path.join(save_folder, '1' + image) # Construct the full image path
  flip.save(save_path)

for image in os.listdir(folder_dir):
  image_path = os.path.join(folder_dir, image) # Construct the full image path
  visual = Image.open(image_path)
  flip = transforms2(visual)
  save_path = os.path.join(save_folder, '2' + image) # Construct the full image path
  flip.save(save_path)

for image in os.listdir(folder_dir):
  image_path = os.path.join(folder_dir, image) # Construct the full image path
  visual = Image.open(image_path)
  flip = transforms3(visual)
  save_path = os.path.join(save_folder, '3' + image) # Construct the full image path
  flip.save(save_path)

for image in os.listdir(folder_dir):
  image_path = os.path.join(folder_dir, image) # Construct the full image path
  visual = Image.open(image_path)
  flip = transforms4(visual)
  save_path = os.path.join(save_folder, '4' + image) # Construct the full image path
  flip.save(save_path)

for image in os.listdir(folder_dir):
  image_path = os.path.join(folder_dir, image) # Construct the full image path
  visual = Image.open(image_path)
  flip = transforms5(visual)
  save_path = os.path.join(save_folder, '5' + image) # Construct the full image path
  flip.save(save_path)

for image in os.listdir(folder_dir):
  image_path = os.path.join(folder_dir, image) # Construct the full image path
  visual = Image.open(image_path)
  flip = transforms6(visual)
  save_path = os.path.join(save_folder, '6' + image) # Construct the full image path
  flip.save(save_path)

for image in os.listdir(folder_dir):
  image_path = os.path.join(folder_dir, image) # Construct the full image path
  visual = Image.open(image_path)
  flip = transforms7(visual)
  save_path = os.path.join(save_folder, '7' + image) # Construct the full image path
  flip.save(save_path)

for image in os.listdir(folder_dir):
  image_path = os.path.join(folder_dir, image) # Construct the full image path
  visual = Image.open(image_path)
  flip = transforms8(visual)
  save_path = os.path.join(save_folder, '8' + image) # Construct the full image path
  flip.save(save_path)

for image in os.listdir(folder_dir):
  image_path = os.path.join(folder_dir, image) # Construct the full image path
  visual = Image.open(image_path)
  flip = transforms9(visual)
  save_path = os.path.join(save_folder, '9' + image) # Construct the full image path
  flip.save(save_path)
  
# save to computer on jupyter notebook
#!zip -r "cdr3_aug_zip.zip" "/content/cdr3_aug"