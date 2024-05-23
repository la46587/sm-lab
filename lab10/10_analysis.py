import cv2
import os
import numpy as np
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def calcMSE(imgOriginal, imgModified):
    return np.mean((imgOriginal.astype("float") - imgModified.astype("float")) ** 2)


def calcNMSE(imgOriginal, imgModified):
    return calcMSE(imgOriginal, imgModified) / np.var(imgOriginal.astype('float'))


def calcPSNR(imgOriginal, imgModified):
    mse = calcMSE(imgOriginal, imgModified)
    maxPixelValue = 255.0 if imgOriginal.dtype == np.uint8 else 1.0
    psnr = 20 * np.log10(maxPixelValue / np.sqrt(mse))
    return psnr


def calcIF(imgOriginal, imgModified):
    mse = calcMSE(imgOriginal, imgModified)
    errorSum = np.sum(mse)
    signalEnergy = np.sum(imgOriginal.astype('float') ** 2)
    imgFidelity = 1 - (errorSum / signalEnergy)
    return imgFidelity


imgOriginal = io.imread('zima/zima.png')
imgOriginal = imgOriginal[..., :3]

imgFake = io.imread('zima/5.jpg')

mse = calcMSE(imgOriginal, imgFake)
print(mse)

nmse = calcNMSE(imgOriginal, imgFake)
print(nmse)

psnr = calcPSNR(imgOriginal, imgFake)
print(psnr)

ifdelity = calcIF(imgOriginal, imgFake)
print(ifdelity)
