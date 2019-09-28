# -*- coding: utf-8 -*-
from skimage import io
import numpy as np, os

from sklearn import model_selection as ms

from sklearn import ensemble

from sklearn.externals import joblib

from extract_features import extract_img_features

#Обозначение путей файлов с картинками
rbc_path = 'training/RBC/'
wbc_path = 'training/WBC/'
platelet_path = 'training/Platelet/'
non_cells_path = 'training/non_cells/'

#Инитиализация X и Y для обучения
Xlist = []
Ylist = []

#Инит. счетчиков
wbc_counter = 0
rbc_counter = 0
platelet_counter = 0

#Извлекаем все признаки картинок RBC
for file in os.listdir(rbc_path):
    print(rbc_path+file)
    
    if file.endswith('.jpg'):
        cur_img = io.imread(rbc_path+file)
    
        fd = extract_img_features(cur_img)

        Xlist.append(fd)
        Ylist.append('RBC')

#Извлекаем все признаки картинок WBC
for file in os.listdir(wbc_path):
    print(wbc_path+file)
    
    if file.endswith('.jpg'):
        cur_img = io.imread(wbc_path+file)    
    
        fd = extract_img_features(cur_img)

        Xlist.append(fd)
        Ylist.append('WBC')

#Извлекаем все признаки картинок Platelet
for file in os.listdir(platelet_path):
    print(platelet_path+file)
    
    if file.endswith('.jpg'):
        cur_img = io.imread(platelet_path+file)    

        fd = extract_img_features(cur_img)

        Xlist.append(fd)
        Ylist.append('Platelet')

#Извлекаем все признаки не клеток
for file in os.listdir(non_cells_path):
    print(non_cells_path+file)

    if file.endswith('.jpg'):
        cur_img = io.imread(non_cells_path+file)
        
        fd = extract_img_features(cur_img)
        
        Xlist.append(fd)
        Ylist.append('Non_cell')

#Выбор классификатора
clf = ensemble.ExtraTreesClassifier(n_estimators=14)
print("Training a  Classifier")
#Вводим X и Y в классификатор
clf.fit(Xlist, Ylist)

#Сохраняем модель
joblib.dump(clf, 'training/bccd_model')
print("Classifier saved to training folder")

#Оценка точности
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = ms.train_test_split(
 Xlist, Ylist, test_size=0.3, random_state=rand_state)

print('Accuracy:', clf.score(X_test, y_test))

