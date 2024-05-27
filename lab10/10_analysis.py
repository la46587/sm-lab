import cv2
import os
import numpy as np
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity
from scipy.stats import pearsonr, spearmanr


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

results = pd.read_csv('results.csv', sep=',')
results.index = [f'subject{i + 1}' for i in range(len(results))]
results = results.drop(columns=['Timestamp', 'Imię/pseudonim:'])
resultsZima = results.drop(columns=['1.1/15', '1.2/15', '1.3/15', '1.4/15', '1.5/15', '1.6/15', '1.7/15', '1.8/15',
                                    '1.9/15', '1.10/15', '1.11/15', '1.12/15', '1.13/15', '1.14/15', '1.15/15',
                                    '2.1/15', '2.2/15', '2.3/15', '2.4/15', '2.5/15', '2.6/15', '2.7/15', '2.8/15',
                                    '2.9/15', '2.10/15', '2.11/15', '2.12/15', '2.13/15', '2.14/15', '2.15/15'])
resultsZima = resultsZima.transpose()

refZima = cv2.imread('zima/zima.png')
refZima = refZima[..., :3]

imagesZima = ['zima/1.jpg', 'zima/2.jpg', 'zima/3.jpg', 'zima/4.jpg', 'zima/5.jpg', 'zima/6.jpg', 'zima/7.jpg',
              'zima/8.jpg', 'zima/9.jpg', 'zima/10.jpg', 'zima/11.jpg', 'zima/12.jpg', 'zima/13.jpg', 'zima/14.jpg',
              'zima/15.jpg']

mseZima = []
nmseZima = []
psnrZima = []
ifZima = []
ssimZima = []

for image in imagesZima:
    image = cv2.imread(image)

    mse = calcMSE(refZima, image)
    mseZima.append(mse)

    nmse = calcNMSE(refZima, image)
    nmseZima.append(nmse)

    psnr = calcPSNR(refZima, image)
    psnrZima.append(psnr)

    IF = calcIF(refZima, image)
    ifZima.append(IF)

resultsZima['MSE'] = mseZima
resultsZima['NMSE'] = nmseZima
resultsZima['PSNR'] = psnrZima
resultsZima['IF'] = ifZima

resultsZimaMeans = []
resultsZimaStds = []

for i in range(1, 16):
    resultsZimaMeans.append(round(resultsZima.loc[f'3.{i}/15'].mean(), 2))
    resultsZimaStds.append(round(resultsZima.loc[f'3.{i}/15'].std(), 2))

resultsZima['Mean'] = resultsZimaMeans
resultsZima['Standard deviation'] = resultsZimaStds

objectiveMeasures = ['MSE', 'NMSE', 'PSNR', 'IF']
correlationResults = {}

for measure in objectiveMeasures:
    pearsonCorr, _ = pearsonr(resultsZima['Mean'], resultsZima[measure])
    spearmanCorr, _ = spearmanr(resultsZima['Mean'], resultsZima[measure])
    correlationResults[measure] = {
        'Pearson': pearsonCorr,
        'Spearman': spearmanCorr
    }

print(correlationResults)
