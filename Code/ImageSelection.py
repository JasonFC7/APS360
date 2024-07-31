import os
import re
from cdr import read_cdr_folders

def read_img_folders(main_folder):
    selected_img = {}
    cdr_list = read_cdr_folders(main_folder)
    for key in cdr_list:
        for disc_folder in os.listdir(main_folder):
            disc_folder_path = os.path.join(main_folder, disc_folder)
            for patient_folder in os.listdir(disc_folder_path):
                if patient_folder.endswith(key + "_MR1") or patient_folder.endswith(key + "_MR2"):
                    patient_folder_path = os.path.join(disc_folder_path, patient_folder)
                    img_folder_path = patient_folder_path + "\PROCESSED\MPRAGE\T88_111"
                    for image in os.listdir(img_folder_path):
                        if image.endswith("mpr_n4_anon_111_t88_masked_gfc_tra_90.gif") or image.endswith("mpr_n3_anon_111_t88_masked_gfc_tra_90.gif") or image.endswith("mpr_n3_anon_111_t88_gfc_tra_90.gif") or image.endswith("mpr_n4_anon_111_t88_gfc_tra_90.gif"):
                            # image_path = os.path.join(img_folder_path, image)
                            selected_img.update({image : [cdr_list[key], img_folder_path]})
                            
    return selected_img
