import pydicom
import os
import numpy as np
import matplotlib.pyplot as plt
from ADNIimageSelection import ImageSelect

slices = []
dicom_files = ImageSelect('ADNI')

for dirpath, dirnames, filenames in os.walk('ADNI'):
        for filename in filenames:
            # if filename.endswith('.dcm') and (counter == 20):
            # if filename.endswith('.dcm'):
            if filename in dicom_files:   
                filepath = os.path.join(dirpath, filename)
                dicom_data = pydicom.dcmread(filepath)
                slices.append(dicom_data)
                print(filepath)
            