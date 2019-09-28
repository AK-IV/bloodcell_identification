# -*- coding: utf-8 -*-
import numpy as np
from skimage.transform import resize
from skimage.feature import hog
from skimage import color

def extract_img_features(img):
    
    #Инитиализация всех особенностей картинки
    features = []
    
    #Сперва уменьшим картинку до 64х64
    img = resize(img, (64, 64))

    #Переведем картинку в серый и YCbCr цветовые пространства
    grey_img = color.rgb2grey(img)
    ycbcr_img = color.rgb2ycbcr(img)

    #Посчитаем HOG цвета Y
    img_hog = hog(ycbcr_img.T[0])

    #Гистограммы
    rhist = np.histogram(img[:,:,0])
    ghist = np.histogram(img[:,:,1])
    bhist = np.histogram(img[:,:,2])

    # Concatenate
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    
    #Spatial binning
    sb = resize(img, (10, 10))

    sb = sb.flatten()
    
    #Сгруппируем все особенности картинки в один вектор
    features = np.concatenate((img_hog, hist_features, sb))
    
    return features
