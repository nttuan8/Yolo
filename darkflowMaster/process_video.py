# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 08:57:20 2018

@author: DELL
"""


import cv2
from darkflow.net.build import TFNet

options = {
            'model'     : 'cfg/yolo.cfg',
            'load'      : 'bin/yolov2.weights',
            'threshold' : 0.3,
            #'gpu'       : 1.0
        }

tfnet = TFNet(options)

capture = cv2.VideoCapture('demo_video/input_video.mp4')

while(capture.isOpened()):
    ret, frame = capture.read()
    if ret:
        result = tfnet.return_predict(frame)
        for box in result:
            tl = (box['topleft']['x'], box['topleft']['y'])
            br = (box['bottomright']['x'], box['bottomright']['y'])
            label = box['label']
            cv2.rectangle(frame, tl, br, (0, 255, 0), 3)
            cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        capture.release()
        cv2.destroyAllWindows()
        break