import cv2
import numpy as np
from scipy.signal import convolve2d as conv
from scipy import ndimage
import math
from matplotlib import pyplot as plt
from numpy.fft import rfft2

def canny(img):
    edges = cv2.Canny(img, 50, 51)
    edges = cv2.add(edges, edges)
    return edges


def getgauss(ksize, sigma):
    kernel = cv2.getGaussianKernel(ksize, sigma, ktype=cv2.CV_32F)
    return kernel


def applyfil(img,kernel):
    filtered = cv2.filter2D(img, cv2.CV_32F, kernel)
    filtered = backtoim(filtered)
    #filtered = conv(img, kernel, mode='same')
    return filtered


def bsobel(im):
    sx = ndimage.sobel(im, axis=0, mode='constant')
    sy = ndimage.sobel(im, axis=1, mode='constant')
    sob = np.hypot(sx, sy)
    return sob


def backtoim(img):
    img1 = np.array(img, dtype=np.uint8)
    img1 = np.resize(img1, (7, 7))
    return img1


def gradientSobel(img):
    xsobel = np.array([1, 2, 1, 0, 0, 0, -1, -2, -1])
    ysobel = np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1])
    xsobel = xsobel.reshape(3, 3)
    ysobel = ysobel.reshape(3, 3)
    grad_x = applyfil(img, xsobel)
    grad_y = applyfil(img, ysobel)
    return grad_x, grad_y


def applyGabor(img,ksize,theta1,freq1):
    kern = cv2.getGaborKernel((ksize, ksize), 1, theta1, freq1, 0.5, 0, ktype=cv2.CV_32F)
    filimg = applyfil(img, kern)
    return filimg


def lap(img):
    lap2 = cv2.Laplacian(img, cv2.CV_32F)
    lap2 = backtoim(lap2)
    return lap2

def sobel(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return sobelx, sobely

def sharpen(img):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    filtered = applyfil(img, kernel)
    return filtered