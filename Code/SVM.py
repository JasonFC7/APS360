from ImageSort import sortbycdr

import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, precision_score
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from PIL import Image

# Likely remove once folder sorting if figured out :)
def load_images(folder):
    images = []
    labels = []
    label_names = sorted(os.listdir(folder))
    for label_folder in label_names:
        label_path = os.path.join(folder, label_folder)
        if not os.path.isdir(label_path):
            continue
        for filename in os.listdir(label_path):
            img_path = os.path.join(label_path, filename)
            try:
                with Image.open(img_path) as img:
                    img = img.convert('L') 
                    img_data = np.asarray(img).flatten()  
                    images.append(img_data)
                    labels.append(label_folder)  
            except Exception as e:
                print(f"Failed to process image {img_path}: {e}")
    return np.array(images), np.array(labels)

folder_path = 'Temp Processed Data'
data, label = load_images(folder_path)

# Encodes labels if string
labelencoder = LabelEncoder()
label = labelencoder.fit_transform(label)

d_train, d_test, l_train, l_test = train_test_split(data, label, test_size = 0.2, random_state = 1)

scaler = StandardScaler()
d_train = scaler.fit_transform(d_train)
d_test = scaler.fit_transform(d_test)

# Testing SVC with linear kernel
svm1 = SVC(kernel = 'linear')
svm1.fit(d_train, l_train)
l_pred1 = svm1.predict(d_test)
print(confusion_matrix(l_test, l_pred1))
print(classification_report(l_test, l_pred1))

import numpy as np
print("Class distribution in training data:", np.bincount(l_train))
print("Class distribution in testing data:", np.bincount(l_test))
