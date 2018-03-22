import cv2
import numpy as np
from scipy.signal import convolve2d as conv
from scipy import ndimage
import math
from allfilter import getgauss, applyfil ,gradientSobel
from matplotlib import pyplot as plt
from numpy.fft import rfft2
import scipy.ndimage

def getblockfreq(img):
    f = np.fft.fft2(img, (7, 7))
    fshift = np.fft.fftshift(f)
    mag = 20 * np.log(np.abs(fshift))
    #mag = mag.max(axis=1)
    #print (mag)
    return mag[3,3]


def getcenterfreq(img):
    ft = rfft2(img, (7, 7))
    #print(ft[3,3])
    return ((9580/49)*ft[3,3])


def freqest(im, orientim, w):
    freq = 0
    rows, cols = np.shape(im)

    # Rotate the image block so that the ridges are vertical
    rotim = scipy.ndimage.rotate(im, orientim, axes=(1, 0), reshape=False, order=3, mode='nearest')

    # Sum down the columns to get a projection of the grey values down
    # the ridges.

    proj = np.sum(rotim, axis=0)
    freq = np.mean(proj)
    return 299792458/freq
