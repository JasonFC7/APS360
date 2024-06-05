import re
import os

def read_cdr(text):
    with open(text, 'r') as file:
        content = file.read()
        
    index = re.search(r'CDR:\s*([\d.]*)', content)
    
    if index:
        cdr_val = index.group(1).strip()
        if cdr_val:
            return cdr_val
        else:
            return None
    else:
        return None
    
def read_cdr_folders(main_folder):
    cdr_val = None
    cdr_list = {}
    
    for disc_folder in os.listdir(main_folder):
        disc_folder_path = os.path.join(main_folder, disc_folder)
        for patient_folder in os.listdir(disc_folder_path):
            patient_folder_path = os.path.join(disc_folder_path, patient_folder)
            for patient_file in os.listdir(patient_folder_path):
                if patient_file.endswith(".txt"):
                    patient_file_path = os.path.join(patient_folder_path, patient_file)
                    if read_cdr(patient_file_path):
                        cdr_val = read_cdr(patient_file_path)
                        name = os.path.basename(patient_file)
                        name = os.path.splitext(name)[0]
                        cdr_list.update({name[5:9]: cdr_val})
                        
    return cdr_list
