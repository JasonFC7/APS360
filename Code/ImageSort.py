from ImageSelection import read_img_folders
import shutil
import os

#cdr1  = CDR value 0 
#cdr2 = CDR value 0.5
#cdr3 = CDR value 1

# selected_img = {image, [cdr_value, image path]}
def sortbycdr():
    selected_img = read_img_folders('Data')
    num = [] # number of images in each folder
    num_1, num_2, num_3 = 0, 0, 0
    for key in selected_img:
        if selected_img.get(key)[0] == '0':
            source = os.path.join(selected_img.get(key)[1], key)
            if key not in 'cdr1':  
                shutil.move(source, 'cdr1')
                num_1 +=1
        elif selected_img.get(key)[0] == '0.5':
            source = os.path.join(selected_img.get(key)[1], key)
            if key not in 'cdr2':  
                shutil.move(source, 'cdr2')
                num_2 +=1
        elif selected_img.get(key)[0] == '1':
            source = os.path.join(selected_img.get(key)[1], key)
            if key not in 'cdr3':  
                shutil.move(source, 'cdr3')
                num_3 +=1
    num.append(num_1)
    num.append(num_2)
    num.append(num_3)
    return num
