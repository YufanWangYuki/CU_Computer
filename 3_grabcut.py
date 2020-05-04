# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:46:02 2020

@author: lenovo
"""
 
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

BACKGROUND_PATH = './columbia.jpg'
IMAGE_PATH = './image.jpg'

# Portrait image
img = cv2.imread(IMAGE_PATH)
# Mask value
mask = np.zeros(img.shape[:2],np.uint8)
# Background model
bgdModel = np.zeros((1,65),np.float64)
# Foreground model
fgdModel = np.zeros((1,65),np.float64)
# Region of interest
rect = (1,1,600,800)

# Use GrabCut
start = time.time()
for i in range(10):
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    print(i)
print((time.time() - start)/100)  

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

cv2.imshow("capture", img)
cv2.imwrite("result_grabcut.jpg", img) 