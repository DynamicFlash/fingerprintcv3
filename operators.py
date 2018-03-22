"""
Title : contains all the operators of image enhancement techniques

cotains : finfill((windows, thetas, freqs, ksize=3)
requirw


"""
import cv2
import numpy as np
from allfilter import getgauss, applyfil ,backtoim, applyGabor,sharpen



def finfil1(windows, thetas, freqs, ksize=3):
    img1 = applyGabor(windows, ksize, thetas, freqs)
    return img1

def normalise(img):
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    return img

def finfil(windows, thetas, freqs, ksize=3):
    nwin = []
    for i in range(0, len(windows)):
        dst = np.ones((8, 8), dtype=np.uint8)
        imgs=[]
        for j in range(0, 7):
            img1 = applyGabor(windows[i], ksize, thetas[i], freqs[i][j])
            imgs.append(img1)

        reduce(lambda x, y: cv2.addWeighted(x, 0.9, y, 0.9, 0), imgs)

        """
        for k in range(0,7):
            if k != 7:
                dst = backtoim(dst)
                dst = cv2.addWeighted(dst, 0.9, imgs[k], 0.9, 0)
            else:
                dst = dst + cv2.addWeighted(imgs[k], 0.9, img[k+1], 0.9, 0)
        """
        nwin.append(dst)

    return nwin

def binarize(img):
    gray = sharpen(img)
    kernal = getgauss(3, -20)
    img2 = applyfil(img, kernal)
    img3 = backtoim(img2)
    th3 = cv2.adaptiveThreshold(img3, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C ,cv2.THRESH_BINARY,11,2 )    #best
    return th3

def binarize2(img):
    gray = sharpen(img)
    #print(type(gray))
    kernal = getgauss(3, -20)
    img2 = applyfil(img, kernal)
    #img2 = cv2.GaussianBlur(gray, (3, 3), -20)
    img3 = backtoim(img2)
    #th3 = cv2.adaptiveThreshold(img3, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 0) #original
    th3 = cv2.adaptiveThreshold(img3, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C ,cv2.THRESH_BINARY,11,2 )    #best
    #blur = cv2.GaussianBlur(img, (5, 5), 0)
    #ret3, th3 = cv2.threshold(img3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #ret, th3 = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    #th3 = cv2.adaptiveThreshold(img3, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 255, 0)


    #print(img2.shape)

    return th3

def binarize3(img):
    img = sharpen(img)
    kernal = getgauss(3,-20)
    img = applyfil(img, kernal)
    img = backtoim(img)
    #th3 = cv2.adaptiveThreshold(img3, 254, cv2.ADAPTIVE_THRESH_GAUSSIAN_C ,cv2.THRESH_BINARY,11,2 )    #best
    th3 = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 0)
    return th3


def binfull(img):
    img2 = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return img2