# -*- coding: utf-8 -*-

print(__doc__)

import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, ensemble
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

from skimage import io

from extract_features import extract_img_features

#Line Width
lw = 2

#Обозначение путей файлов с картинками
rbc_path = 'training/RBC/'
wbc_path = 'training/WBC/'
platelet_path = 'training/Platelet/'
non_cells_path = 'training/non_cells/'

#Инитиализация X и Y для обучения
Xlist = []
Ylist = []

#Извлекаем все признаки картинок RBC
for file in os.listdir(rbc_path):
    
    if file.endswith('.jpg'):
        cur_img = io.imread(rbc_path+file)
    
        fd = extract_img_features(cur_img)

        Xlist.append(fd)
        Ylist.append('RBC')

#Извлекаем все признаки картинок WBC
for file in os.listdir(wbc_path):
    
    if file.endswith('.jpg'):
        cur_img = io.imread(wbc_path+file)    
    
        fd = extract_img_features(cur_img)

        Xlist.append(fd)
        Ylist.append('WBC')

#Извлекаем все признаки картинок Platelet
for file in os.listdir(platelet_path):

    if file.endswith('.jpg'):
        cur_img = io.imread(platelet_path+file)    

        fd = extract_img_features(cur_img)

        Xlist.append(fd)
        Ylist.append('Platelet')

#Извлекаем все признаки не клеток
for file in os.listdir(non_cells_path):

    if file.endswith('.jpg'):
        cur_img = io.imread(non_cells_path+file)
        
        fd = extract_img_features(cur_img)
        
        Xlist.append(fd)
        Ylist.append('Non_cell')

#Завершено
print("Done extracting features")

#Переводим X и Y в NumPy array
X = np.asarray(Xlist)
y = np.asarray(Ylist)

#Классы
cell_classes = ['RBC', 'WBC', 'Platelet', 'Non_cell']

# Binarize the output
y = label_binarize(y, cell_classes)
n_classes = y.shape[1]

# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, n_features)]

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(ensemble.RandomForestClassifier(random_state=random_state,
                                                                     n_jobs=-1,
                                                                     n_estimators = 14))
y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(figsize=(15,10))
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(cell_classes[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontsize=16)
plt.ylabel('True Positive Rate',fontsize=16)
plt.title('ROC для ExtraTreesClassifier',fontsize=20)
plt.legend(loc="lower right", fontsize=16)
plt.show()