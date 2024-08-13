import matplotlib.pyplot as plt
import csv
from ADNIimageSelection import ImageSelect
from ADNIsliceSelection import finalSlices
import os
import shutil

dicom_files = ImageSelect('ADNI 3')
slice_files = []

for dirpath, dirnames, filenames in os.walk('ADNI 3'):
    for filename in filenames:
        if filename in dicom_files:   
            filepath = os.path.join(dirpath, filename)
            slice_files.append(filepath)
                
filename = "Code/idaSearch_8_13_2024 sag.csv"

cols, rows = [], []

cdr1, cdr2, cdr3 = [], [], []

with open(filename, 'r') as csvfile:
    readcsv = csv.reader(csvfile)
    cols = next(readcsv)
    for row in readcsv:
        rows.append(row)
        
for row in rows:
    if row[3] == '0.0' and row[4] != 'MP-RAGE REPEAT':
        cdr1.append(row[0])
    elif row[3] == '0.5' and row[4] != 'MP-RAGE REPEAT':
        cdr2.append(row[0])
    elif row[3] == '1.0' and row[4] != 'MP-RAGE REPEAT':
        cdr3.append(row[0])

num_1 = 0
num_2 = 0
num_3 = 0

# Not sure... but everytime this is run, the file names change slightly... so it adds the image again... so please delete extras...

for patient in cdr1:
    for slicef in slice_files:
        if patient in slicef:
            fname = os.path.basename(slicef)
            if os.path.isfile('Final ADNI test set/cdr1/{}'.format(fname)):
                None
            else:
                shutil.move(slicef, 'Final ADNI test set/cdr1')
                num_1 += 1
                break
            
for patient in cdr2:
    for slicef in slice_files:
        if patient in slicef:
            fname = os.path.basename(slicef)
            if os.path.isfile('Final ADNI test set/cdr2/{}'.format(fname)):
                None
            else:
                shutil.move(slicef, 'Final ADNI test set/cdr2')
                num_2 += 1
                break

for patient in cdr3:
    for slicef in slice_files:
        if patient in slicef:
            fname = os.path.basename(slicef)
            if os.path.isfile('Final ADNI test set/cdr3/{}'.format(fname)):
                None
            else:
                shutil.move(slicef, 'Final ADNI test set/cdr3')
                num_3 += 1
                break
            
print("Images in cdr1:", num_1)
print("Images in cdr2:", num_2)
print("Images in cdr3:", num_3)

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