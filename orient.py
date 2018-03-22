"""
calculateOrientation2(image, w)

gets orientaton of for the image block image


"""


import numpy as np
import math
from allfilter import getgauss, applyfil ,gradientSobel,canny
import theimgseg



def calculateOrientation21(image, w):

    # smooth image with gaussian blur
    kernel = getgauss(3, 5)
    smoothed_im = applyfil(image, kernel)

    # calculates gradients with sobel filter
    dx, dy = gradientSobel(smoothed_im)

    # smooth gradients
    Gx = applyfil(dx, kernel)
    Gy = applyfil(dy, kernel)

    # compute gradient magnitude
    vx = 2*Gx * Gy
    vy = (Gx**2)-(Gy ** 2)
    #G = np.sqrt(Gxx + Gyy)

    # calculate theta
    theta = 0.5*np.arctan2(vx, vy)

    # smoothens the angles

    tabx =  np.zeros((w, w), np.uint8)
    taby =  np.zeros((w, w), np.uint8)
    for p in range(0, w-1):
        for q in range(0, w-1):
            tabx[p, q] = math.cos(2*theta[p, q])
            taby[p, q] = math.sin(2*theta[p, q])
            theta[p, q] = np.pi + math.atan2(tabx[p, q], taby[p, q])

    return theta[w / 2, w / 2]


def calculateOrientation2(image, w):
    # smooth image with gaussian blur
    k = 0
    kernel = getgauss(5, 5)
    smoothed_im = applyfil(image, kernel)

    # calculates gradients with sobel filter
    dx, dy = gradientSobel(smoothed_im)

    # smooth gradients
    Gx = applyfil(dx, kernel)
    Gy = applyfil(dy, kernel)

    # compute gradient magnitude
    vx = 2 * Gx * Gy
    vy = (Gx ** 2) - (Gy ** 2)
    # G = np.sqrt(Gxx + Gyy)


    # calculate theta
    theta = 0.5 * np.arctan2(vx, vy)
    print(theta)
    # smoothens the angles

    tabx = np.zeros((w, w), np.uint8)
    taby = np.zeros((w, w), np.uint8)
    for p in range(0, w - 1):
        for q in range(0, w - 1):
            tabx[p, q] = math.cos(2 * theta[p, q])
            taby[p, q] = math.sin(2 * theta[p, q])

            if (theta[p, q] < 0 and vx[p,q] < 0) or (vx[p,q]>0 and theta[p,q]> 0 ):
                k =0.5

            if (theta[p, q] < 0 and vx[p,q] > 0):
                k =1

            if (theta[p, q] > 0 and vx[p,q] < 0):
                k =0

            theta[p, q] = k*np.pi + math.atan2(tabx[p, q], taby[p, q])

    return theta[w / 2, w / 2]

def getedges(img):
    equ= canny(img)
    return img

def calculateOrientation31(image, w):
    image = getedges(image)

    """"""
    # smooth image with gaussian blur
    k = 0
    kernel = getgauss(5, 5)
    smoothed_im = applyfil(image, kernel)

    # calculates gradients with sobel filter
    dx, dy = gradientSobel(smoothed_im)

    # smooth gradients
    Gx = applyfil(dx, kernel)
    Gy = applyfil(dy, kernel)

    # compute gradient magnitude
    vx = 2 * Gx * Gy
    vy = (Gx ** 2) - (Gy ** 2)
    # G = np.sqrt(Gxx + Gyy)


    # calculate theta
    theta = 0.5 * np.arctan2(vx, vy)
    print(theta)
    # smoothens the angles

    tabx = np.zeros((w, w), np.uint8)
    taby = np.zeros((w, w), np.uint8)
    for p in range(0, w - 1):
        for q in range(0, w - 1):
            tabx[p, q] = math.cos(2 * theta[p, q])
            taby[p, q] = math.sin(2 * theta[p, q])

            if (theta[p, q] < 0 and vx[p,q] < 0) or (vx[p,q]>0 and theta[p,q]> 0 ):
                k =0.5

            if (theta[p, q] < 0 and vx[p,q] > 0):
                k =1

            if (theta[p, q] > 0 and vx[p,q] < 0):
                k =0

            theta[p, q] = k*np.pi + math.atan2(tabx[p, q], taby[p, q])

    return theta[w / 2, w / 2]

def calculateOrientation3(image, w):
    """
    :param image: image segment
    :param w: blocksize
    :return: block orientation
    """
    # smooth image with gaussian blur
    kernel = getgauss(5, 5)
    smoothed_im = applyfil(image, kernel)

    # calculates gradients with sobel filter
    dx, dy = gradientSobel(smoothed_im)

    # smooth gradients
    Gx = applyfil(dx, kernel)
    Gy = applyfil(dy, kernel)
    vx = vy =0
    # compute gradient magnitude
    for p in range(0, w - 1):
        for q in range(0, w - 1):

            if Gx[p, q]== 0 :
                Gx[p, q] = 1

            if Gy[p, q]== 0:
                Gy[p,q]= 1

            if Gx[p, q]== 0 and Gy[p, q]== 0 :
                Gx[p, q] = Gy[p, q] = 1


            vx += 2 * Gx[p, q] * Gy[p, q]
            vy += (Gx[p, q] ** 2) - (Gy[p, q] ** 2)
    # G = np.sqrt(Gxx + Gyy)


    # calculate theta
    theta = 0.5 * np.arctan2(vx, vy)
   # print(theta)
    # smoothens the angles

    if (theta< 0 and vx< 0) or (vx > 0 and theta> 0):
        k = 0.5

    if (theta < 0 and vx > 0):
        k = 1

    if (theta > 0 and vx < 0):
        k = 0

    tabx = math.cos(2 * theta)
    taby = math.sin(2 * theta)

    theta= k * np.pi + math.atan2(tabx, taby)


    return theta*180/np.pi