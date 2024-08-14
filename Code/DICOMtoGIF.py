import pydicom
from PIL import Image
import numpy as np
import os

cdr1_path = 'Final ADNI test set (DCM)/cdr3'

for root, dirs, files in os.walk(cdr1_path):
    for file in files:                
        fpath = os.path.join(root, file)
        
        dicom = pydicom.dcmread(fpath)
        img_array = dicom.pixel_array
        normalized_img = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array)) * 255
        normalized_img = normalized_img.astype(np.uint8)
        final_img = Image.fromarray(normalized_img)
        
        fname = os.path.splitext(file)[0]
        gif_name = fname + '.gif'

        final_img.save(gif_name, 'GIF')