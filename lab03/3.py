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

def imgResize(img, scale):
    x = img.shape[1]
    y = img.shape[0]

    X = np.ceil(x * scale).astype(int)
    Y = np.ceil(y * scale).astype(int)

    imgResized = np.zeros((X, Y, 3), dtype=np.uint8)

    xx = np.linspace(0, x - 1, X)
    yy = np.linspace(0, y - 1, Y)

    for i in range(0, X):
        for j in range(0, Y):
            imgResized[i, j] = img[np.ceil(xx[i]).astype(int), np.ceil(yy[j]).astype(int)]

    return imgResized

img = plt.imread('SMALL_0002.png')
plt.imshow(img)
plt.show()

img = imgToUInt8(img)
newImg = imgResize(img, 0.01)
plt.imshow(newImg)
plt.show()
