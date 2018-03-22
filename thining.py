import cv2
import numpy as np

def thinit2(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    #erosion = cv2.erode(img, kernel, iterations=1)
    #erosion = cv2.bitwise_not(img)
    img_erosion = cv2.erode(img, kernel, iterations=1)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
    temp = cv2.bitwise_not(img_erosion)
    #temp = cv2.subtract(img, img_erosion)
    #closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return temp

