import numpy as np
from orient import calculateOrientation3,calculateOrientation2
from freqency import getcenterfreq, getblockfreq, freqest
from operators import finfil1, binarize, binarize3
from allfilter import lap, sobel, sharpen, backtoim
from orient import getedges
import cv2



def applyall(img):
    rows, cols = img.shape
    imgfil=np.zeros((rows, cols),   dtype=np.uint8)
    w=7
    winb=np.resize(imgfil, (7, 7))
    print(winb.shape)


    for l in range(0, rows-1,w):
        for m in range(0, cols-1, w):

            angel = np.degrees(calculateOrientation3(img[l: l + w, m:m + w], w))
            #freq  = getblockfreq(img[l: l+w, m:m+w])
            freq  = freqest(img[l: l+w, m:m+w],angel,w)
            imgfil[l:l + w, m:m + w] = finfil1(img[l: l+w, m:m+w], angel, freq)


    return imgfil


def applylap(img):
    rows, cols = img.shape
    imgfil=np.zeros((rows, cols),   dtype=np.uint8)
    w = 7
    winb=np.resize(imgfil, (7, 7))
    print(winb.shape)


    for l in range(0, rows-1,w):
        for m in range(0, cols-1, w):
            imgfil[l:l + w, m:m + w] = lap(img[l:l + w, m:m + w])

    return imgfil

def applysobel(img):
    rows, cols = img.shape
    imgfil = np.zeros((rows, cols), dtype=np.uint8)
    w = 7
    winb = np.resize(imgfil, (7, 7))
    print(winb.shape)

    for l in range(0, rows - 1, w):
        for m in range(0, cols - 1, w):
            imgfil[l:l + w, m:m + w] = sobel(img[l:l + w, m:m + w])

    return imgfil

def binseg(img):
    rows, cols = img.shape
    imgfil = np.zeros((rows, cols), dtype=np.uint8)
    w = 7
    winb = np.resize(imgfil, (7, 7))
    print(winb.shape)

    for l in range(0, rows - 1, w):
        for m in range(0, cols - 1, w):
            imgfil[l:l + w, m:m + w] = sobel(img[l:l + w, m:m + w])

    return imgfil

def applybin(img):
    rows, cols = img.shape
    imgfil = np.zeros((rows, cols), dtype=np.uint8)
    w = 7
    winb = np.resize(imgfil, (7, 7))
    print(winb.shape)

    for l in range(0, rows - 1, w):
        for m in range(0, cols - 1, w):
            imgfil[l:l + w, m:m + w] = binarize(img[l:l + w, m:m + w])

    return imgfil

def applybin2(img):
    rows, cols = img.shape
    imgfil = np.zeros((rows, cols), dtype=np.uint8)
    w = 7
    winb = np.resize(imgfil, (7, 7))
    print(winb.shape)

    for l in range(0, rows - 1, w):
        for m in range(0, cols - 1, w):
            imgfil[l:l + w, m:m + w] = binarize3(img[l:l + w, m:m + w])

    return imgfil

def alledges(img):
    rows, cols = img.shape
    imgfil=np.zeros((rows, cols),   dtype=np.uint8)
    w=7
    winb=np.resize(imgfil, (7, 7))
    for l in range(0, rows-1,w):
        for m in range(0, cols-1, w):
            imgfil[l:l + w, m:m + w] = getedges(img[l: l+w, m:m+w])


    return imgfil


def binfull(img):
    img2 = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return img2

def edgesfull(img):
    edges = cv2.Canny(img, 50, 51,L2gradient=True)
    edges = cv2.add(edges, edges)
    return edges