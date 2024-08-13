import os
import pydicom

def ImageSelect(data):
    root_dir = data

    dicom_files = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        counter = 0
        breaker = 0
        if len(filenames) > 1:
            filenames = sorted(filenames, key=lambda x: int(x.split('_')[-3]))
            for file in filenames:
                counter += 1
                index = int(len(filenames) * 0.55)
                if file.endswith('.dcm') and (counter == index) and (breaker == 0):
                    dicom_files.append(file)
                    breaker += 1

    return dicom_files
        

# root_dir = 'ADNI 4'

# dicom_files = []

# for dirpath, dirnames, filenames in os.walk(root_dir):
#     counter = 0
#     if len(filenames) > 1:
#         filenames = sorted(filenames, key=lambda x: int(x.split('_')[-3]))
#         for file in filenames:
#             counter += 1
#             index = len(filenames) / 3
#             if file.endswith('.dcm') and (counter == index):
#                 dicom_files.append(os.path.join(dirpath, file))

# for filename in dicom_files:
#     print(filename)