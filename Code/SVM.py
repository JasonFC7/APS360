from ImageSort import sortbycdr

import os
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split,GridSearchCV
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

folder_path = 'Processed Images'
data, label = load_images(folder_path)

# Encodes labels if string
labelencoder = LabelEncoder()
label = labelencoder.fit_transform(label)

d_train, d_test, l_train, l_test = train_test_split(data, label, test_size = 0.2, random_state = 1)

scaler = StandardScaler()
d_train = scaler.fit_transform(d_train)
d_test = scaler.fit_transform(d_test)

print("Class distribution in training data:", np.bincount(l_train))
print("Class distribution in testing data:", np.bincount(l_test))

C = 1.0

# Testing Linear SVC with linear kernel
svm2 = LinearSVC(C=C, max_iter=1000)
svm2.fit(d_train, l_train)
l_pred2 = svm2.predict(d_test)
print('----------Result of linear SVC----------')
print(confusion_matrix(l_test, l_pred2))
print(classification_report(l_test, l_pred2))

#
# # Testing SVC with linear kernel
# svm1 = SVC(kernel = 'linear', C=C)
# svm1.fit(d_train, l_train)
# l_pred1 = svm1.predict(d_test)
# print(confusion_matrix(l_test, l_pred1))
# print(classification_report(l_test, l_pred1))
#
# # Testing SVC with RBF kernel
# svm3 = SVC(kernel="rbf", gamma=0.7, C=C)
# svm3.fit(d_train, l_train)
# l_pred3 = svm3.predict(d_test)
# print(confusion_matrix(l_test, l_pred3))
# print(classification_report(l_test, l_pred3))
#
# # Testing SVC with polynomial
# svm4 = SVC(kernel="poly", degree=2, gamma="auto", C=C)
# svm4.fit(d_train, l_train)
# l_pred4 = svm4.predict(d_test)
# print(confusion_matrix(l_test, l_pred4))
# print(classification_report(l_test, l_pred4))

param_grid_linear = {'C': [0.1, 1, 10, 100]}
grid_linear = GridSearchCV(SVC(kernel='linear'), param_grid_linear, cv=5, scoring='accuracy')
grid_linear.fit(d_train, l_train)
print("Best parameters for Linear kernel:", grid_linear.best_params_)
print("Best accuracy for Linear kernel:", grid_linear.best_score_)
print()
linear_pred = grid_linear.predict(d_test)
print("Linear kernel - Accuracy on Test data:", accuracy_score(l_test, linear_pred))
print(confusion_matrix(l_test, linear_pred))
print(classification_report(l_test, linear_pred))

param_grid_poly = {'C': [0.1, 1, 10, 100], 'degree': [2, 3, 4], 'gamma': [0.1, 1, 'scale', 'auto']}
grid_poly = GridSearchCV(SVC(kernel='poly'), param_grid_poly, cv=5, scoring='accuracy')
grid_poly.fit(d_train, l_train)
print("Best parameters for Poly kernel:", grid_poly.best_params_)
print("Best accuracy for Poly kernel:", grid_poly.best_score_)
print()
poly_pred = grid_poly.predict(d_test)
print("Poly kernel - Accuracy on Test data:", accuracy_score(l_test, poly_pred))
print(confusion_matrix(l_test, poly_pred))
print(classification_report(l_test, poly_pred))

param_grid_rbf = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 1, 'scale', 'auto']}
grid_rbf = GridSearchCV(SVC(kernel='rbf'), param_grid_rbf,cv=5, scoring='accuracy')
grid_rbf.fit(d_train, l_train)
print("Best parameters for RBF kernel:", grid_rbf.best_params_)
print("Best accuracy for RBF kernel:", grid_rbf.best_score_)
print()
rbf_pred = grid_rbf.predict(d_test)
print("RBF kernel - Accuracy on Test data:", accuracy_score(l_test, rbf_pred))
print(confusion_matrix(l_test, rbf_pred))
print(classification_report(l_test, rbf_pred))


