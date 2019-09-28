# -*- coding: utf-8 -*-

import numpy as np

def heatmap(image, detections, threshold):
    
    #прибавить +10 к каждому пикселю выделения
    heatmap = np.zeros_like(image)
    for box in detections:
        heatmap[box[0][1]:box[1][1],
                box[0][0]:box[1][0]] += 10
    
    heatmap[heatmap <= threshold] = 0
    
    return heatmap