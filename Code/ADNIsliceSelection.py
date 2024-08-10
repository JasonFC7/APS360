import pydicom
import os
import numpy as np
import matplotlib.pyplot as plt
from ADNIimageSelection import ImageSelect

def finalSlices(dicom_dir):
    dicom_files = ImageSelect(dicom_dir)

    for i in range(len(dicom_files)):
        print(dicom_files[i])
    print(len(dicom_files))

    slices = []

    for dirpath, dirnames, filenames in os.walk(dicom_dir):
        for filename in filenames:
            # if filename.endswith('.dcm') and (counter == 20):
            # if filename.endswith('.dcm'):
            if filename in dicom_files:   
                filepath = os.path.join(dirpath, filename)
                dicom_data = pydicom.dcmread(filepath)
                slices.append(dicom_data)

    slices.sort(key=lambda x: int(x.InstanceNumber))
    images = [s.pixel_array for s in slices]

    max_height = max(img.shape[0] for img in images)
    max_width = max(img.shape[1] for img in images)

    images_padded = []
    for img in images:
        pad_height = max_height - img.shape[0]
        pad_width = max_width - img.shape[1]

        padded_img = np.pad(img, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)
        images_padded.append(padded_img)

    img_slices = np.stack(images_padded, axis=0)

    img_slices = img_slices.astype(np.float32)
    img_slices /= np.max(img_slices)

    # for i in range(img_slices.shape[0]):
    #     plt.figure(figsize=(5, 5))
    #     plt.imshow(img_slices[i], cmap='gray')
    #     plt.title(f'Slice {i + 1}')
    #     plt.axis('off')
    #     plt.show()

    return img_slices