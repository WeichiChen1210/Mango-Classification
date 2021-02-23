"""
Preprocessing.
"""
import os
import cv2
import numpy as np

# kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) # kernel for sharpening
# kernel = np.array([[-0.125, -0.125, -0.125, -0.125, -0.125],
#                    [-0.125, 0.25, 0.25, 0.25, -0.125], 
#                    [-0.125, 0.25, 1.0, 0.25, -0.125], 
#                    [-0.125, 0.25, 0.25, 0.25, -0.125],
#                    [-0.125, -0.125, -0.125, -0.125, -0.125]]) # kernel for sharpening
sharpen_kernel = np.array([[1, 4, 6, 4, 1],
                   [4, 16, 24, 16, 4],
                   [6, 24, -476, 24, 6], 
                   [4, 16, 24, 16, 4],
                   [1, 4, 6, 4, 1],]) * (-1 / 256)# kernel for sharpening

def adjust_saturation(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    if value < 1 and value > 0:
        hsv[:, :, 1] = saturation + saturation * value
    elif value > 1:
        hsv[:, :, 1] = saturation + value
    hsv[hsv > 255] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr

def sharpen(img, kernel=None):
    if kernel is None:
        kernel = sharpen_kernel
    sharp = cv2.filter2D(img, -1, kernel)
    sharp[sharp > 255] = 255

    return sharp

def histogram_equalization(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])

    result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    return result