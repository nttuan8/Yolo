# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 08:06:46 2018

@author: DELL
"""

import cv2
import matplotlib.pyplot as plt

from darkflow.net.build import TFNet

options = {
            'model'     : 'cfg/yolo.cfg',
            'load'      : 'bin/yolov2.weights',
            'threshold' : 0.3,
            #'gpu'       : 1.0
        }

tfnet = TFNet(options)

img = cv2.imread('cat.jpg')
result = tfnet.return_predict(img)

for box in result:
    tl = (box['topleft']['x'], box['topleft']['y'])
    br = (box['bottomright']['x'], box['bottomright']['y'])
    label = box['label']
    cv2.rectangle(img, tl, br, (0, 255, 0), 3)
    cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)

cv2.imshow('img', img)