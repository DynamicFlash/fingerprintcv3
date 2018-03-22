import cv2

from operators import normalise,binarize3,binfull
from theimgseg import applyall, applylap, applybin,applybin2,edgesfull
from matplotlib import pyplot as plt
from freqency import freqest
from thining import thinit2



if __name__ == "__main__":

    img = cv2.imread("1.jpg", 0)
    img = cv2.resize(img,(259, 259))
    img = normalise(img)


    imgfil1 = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    imgfil2 = applyall(imgfil1)
    imgfil3 = cv2.bitwise_not(imgfil2)
    imgfil3 = thinit2(imgfil3)
    imgfil3 = cv2.adaptiveThreshold(imgfil3, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)




    i=j= 100
    plt.subplot(131),plt.imshow(img, 'gray'),plt.title("Original")
    plt.subplot(132), plt.imshow(imgfil2,cmap='gray'),plt.title("after gabor")
    plt.subplot(133), plt.imshow(imgfil3, 'gray'),plt.title("filtered"), plt.show()
