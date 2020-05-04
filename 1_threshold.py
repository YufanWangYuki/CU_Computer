# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 07:42:38 2019

@author: lenovo
"""
from PIL import Image
import numpy as np
import cv2
import time

def fillHole(im_in):
	im_floodfill = im_in.copy()

	# Mask used to flood filling.
	# Notice the size needs to be 2 pixels than the image.
	h, w = im_in.shape[:2]
	mask = np.zeros((h+2, w+2), np.uint8)

	# Floodfill from point (0, 0)
	cv2.floodFill(im_floodfill, mask, (0,0), 255);

	# Invert floodfilled image
	im_floodfill_inv = cv2.bitwise_and(im_floodfill)

	# Combine the two images to get the foreground.
	im_out = im_in | im_floodfill_inv

	return im_out


cap = cv2.VideoCapture(0)
cap.set(5,10)
 
# Target output
img_back = cv2.imread('back.jpg')
i = 0
sum_value = 0
 
while (True):
    start = time.time()
    ret,frame = cap.read()
    if ret == False:
        continue
    
    # Get size
    rows, cols, channels = frame.shape
 
    lower_color = np.array([150, 150, 150])
    upper_color = np.array([255, 255, 255])
    
    # Make mask
    fgmask = cv2.inRange(frame, lower_color, upper_color)
    
    cv2.imshow('Mask', fgmask)
    sum_value = sum_value + (time.time() - start)
    
    
    erode = cv2.erode(fgmask, None, iterations=1)
    cv2.imshow('erode', erode)
    dilate = cv2.dilate(erode, None, iterations=1)
    cv2.imshow('dilate', dilate)
 
    rows, cols = dilate.shape
    img_back=img_back[0:rows,0:cols]

    img2_fg = cv2.bitwise_and(img_back, img_back, mask=dilate)
    Mask_inv = cv2.bitwise_not(dilate)
    img3_fg = cv2.bitwise_and(frame, frame, mask=Mask_inv)
    
    finalImg=img2_fg + img3_fg
#    finalImg = fillHole(finalImg)
    cv2.imshow('res', finalImg)
 
    k = cv2.waitKey(10) & 0xFF
    i += 1
    print(sum_value/(i+1))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
 
cv2.destroyAllWindows()