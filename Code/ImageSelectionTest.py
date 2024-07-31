import os

root_dir = 'ADNI 3'

dicom_files = []

for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        if file.endswith('.dcm'):
            dicom_files.append(os.path.join(dirpath, file))

# Optional: Sort the DICOM files by a specific criterion (e.g., by instance number or filename)
# Here we assume the filenames contain instance numbers that can be sorted
dicom_files.sort()

# Print out the collected DICOM file paths
for dicom_file in dicom_files:
    print(dicom_file)