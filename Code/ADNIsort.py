import matplotlib.pyplot as plt
import csv
from ADNIimageSelection import ImageSelect
from ADNIsliceSelection import finalSlices
import os
import shutil

dicom_files = ImageSelect('ADNI')
slice_files = []

for dirpath, dirnames, filenames in os.walk('ADNI'):
    for filename in filenames:
        if filename in dicom_files:   
            filepath = os.path.join(dirpath, filename)
            slice_files.append(filepath)
                
filename = "idaSearch_7_31_2024.csv"

cols, rows = [], []

cdr1, cdr2, cdr3 = [], [], []

with open(filename, 'r') as csvfile:
    readcsv = csv.reader(csvfile)
    cols = next(readcsv)
    for row in readcsv:
        rows.append(row)
        
for row in rows:
    if row[3] == '0.0':
        cdr1.append(row[0])
    elif row[3] == '0.5':
        cdr2.append(row[0])
    elif row[3] == '1.0':
        cdr3.append(row[0])

num_1 = 0
num_2 = 0
num_3 = 0

for patient in cdr1:
    for filepath in slice_files:
        if patient in filepath:
            if filepath not in 'Final ADNI test set\cdr1':
                 shutil.move(filepath, 'Final ADNI test set\cdr1')
                 num_1 +=1
            break

print(num_1)

# for filepath in slice_files:
#     print(filepath)
#     for patient in cdr1:
#         print(patient)
    #     if patient in filepath:
    #         if filepath not in 'Final ADNI test set\cdr1':
    #             shutil.move(filepath, 'Final ADNI test set\cdr1')
    #             num_1 +=1
    # for patient in cdr2:
    #     if patient in filepath:
    #         if filepath not in 'Final ADNI test set\cdr2':
    #             shutil.move(filepath, 'Final ADNI test set\cdr2')
    #             num_2 +=1
    #         print(filepath, 'in')
    # for patient in cdr3:
    #     if patient in filepath:
    #         if filepath not in 'Final ADNI test set\cdr3':
    #             shutil.move(filepath, 'Final ADNI test set\cdr3')
    #             num_3 +=1
        
# print(num_1)
# print(num_2)
# print(num_3)