# -*- coding: utf-8 -*-

#
#
# ФАЙЛ ДЛЯ ОТЛАДКИ
#
#

# Import the required modules
from skimage import io

from sklearn.externals import joblib

import numpy as np
import time

from heatmap import heatmap
from nms import nms

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from extract_features import extract_img_features


def sliding_window(image, window_size, step_size):
    '''
    This function returns a patch of the input image `image` of size equal
    to `window_size`. The first image returned top-left co-ordinates (0, 0) 
    and are increment in both x and y directions by the `step_size` supplied.
    So, the input parameters are -
    * `image` - Input Image
    * `window_size` - Size of Sliding Window
    * `step_size` - Incremented Size of Window

    The function returns a tuple -
    (x, y, im_window)
    where
    * x is the top-left x co-ordinate
    * y is the top-left y co-ordinate
    * im_window is the sliding window image
    '''
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


img_path = 'BCCD/cells/_0_1106.jpeg'

show_img = io.imread(img_path, as_grey=False)
image = io.imread(img_path, as_grey=False)

min_wdw_sz = (64, 64)
step_size = (8, 8)

fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 6))
ax = ax.ravel()

t1 = time.time()

# Load the classifier
clf = joblib.load('training/bccd_model')

# List to store the detections
detections = []
img_heatmap = []

# If the width or height of the scaled image is less than
# the width or height of the window, then end the iterations.
for (x, y, im_window) in sliding_window(image, min_wdw_sz, step_size):
    if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
        continue

    fd = extract_img_features(im_window)

    fd = [fd]

    pred = clf.predict(fd)
    predict_probability = np.amax(clf.predict_proba(fd))

    if pred[0] != 'Non_cell':
        print('DETECTION!', pred[0])

        if pred[0] == 'WBC':
            cell_color = 'blue'
        elif pred[0] == 'RBC':
            cell_color = 'red'
        else:
            cell_color = 'green'

        box = [[x, y], [x + int(min_wdw_sz[0]), y + int(min_wdw_sz[1])]]
        img_heatmap.append(box)
        # print(clf.decision_function(fd))
        detections.append((x, y, np.amax(predict_probability),
                           min_wdw_sz[0], min_wdw_sz[1], pred[0]))

        rect = mpatches.Rectangle((x, y), im_window.shape[0],
                                  im_window.shape[1],
                                  fill=False, edgecolor=cell_color, linewidth=1)
        ax[0].add_patch(rect)

t2 = time.time()

print('Exec time: ', (t2-t1))

# Perform Non Maxima Suppression
detections = nms(detections, 0.21)

#HEATMAP
heat = heatmap(image, img_heatmap, 1)

# Display the results after performing NMS
for (x_tl, y_tl, _, w, h, cell_label) in detections:
    
    # Draw the detections
    if cell_label == "RBC":
        cell_color = 'red'
    elif cell_label == "WBC":
        cell_color = 'blue'
    elif cell_label == "Platelet":
        cell_color = 'green'

    rect = mpatches.Rectangle((x_tl, y_tl), w, h, fill=False,
                              edgecolor=cell_color, linewidth=1.25)
    ax[2].add_patch(rect)
    ax[2].text(x_tl, y_tl - 5, cell_label)

ax[2].imshow(show_img)

ax[1].imshow(heat)

ax[0].imshow(show_img)
