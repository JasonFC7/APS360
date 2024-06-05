from PIL import Image
import os
import re

def read_img_folders(main_folder):
    selected_img = {}
    read_cdr_folders(main_folder)
    for key in cdr_list:
        for disc_folder in os.listfir(main_folder):
            disc_folder_path = os.path.join(main_folder, disc_folder)
            for patient_folder in os.listdir(disc_folder_path):
                if patient_folder.endswith(key,"_MR1"):
                    patient_folder_path = os.path.join(disc_folder_path, patient_folder)
                    img_folder_path = patient_folder_path + "\PROCESSED\MPRAGE\T88_111"
                    for image in os.listdir(img_folder_path):
                        if image.endswith("MR1_mpr_n4_anon_111_t88_masked_gfc_tra_90.gif"):
                            selected_img.update({image: cdr_list[key]})
                            
    return selected_img