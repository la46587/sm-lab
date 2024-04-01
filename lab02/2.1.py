import numpy as np
import matplotlib.pyplot as plt
import cv2


def imgToUInt8(img):
    if np.issubdtype(img.dtype, np.integer) or np.issubdtype(img.dtype, np.unsignedinteger):
        return img
    else:
        img = img * 255.0
        img = img.astype('uint8')
        return img


def imgToFloat(img):
    if np.issubdtype(img.dtype, np.floating):
        return img
    else:
        img = img / 255.0
        return img


img1 = cv2.imread('B01.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
plt.imshow(img1)
plt.show()

R = img1[:, :, 0]
G = img1[:, :, 1]
B = img1[:, :, 2]

Y1 = 0.299 * R + 0.587 * G + 0.114 * B
Y2 = 0.2126 * R + 0.7152 * G + 0.0722 * B

plt.imshow(Y2, cmap=plt.cm.gray)
plt.show()
