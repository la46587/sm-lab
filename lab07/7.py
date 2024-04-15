from docx import Document
from docx.shared import Inches
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import sounddevice as sd
import soundfile as sf


def quantize(signal, targetBits):
    targetMax = 2 ** (targetBits - 1)
    quantizedSignal = np.round(signal / targetMax) * targetMax
    return quantizedSignal


def encodeALaw(x, quantizationLevel, A=87.6):
    signs = np.sign(x)
    compression = np.log(1 + A * np.abs(x)) / np.log(1 + A)
    quantized = np.round(compression * (2 ** quantizationLevel - 1))
    return signs * quantized


def decodeALaw(y, quantizationLevel, A=87.6):
    signs = np.sign(y)
    expansion = (np.exp(np.abs(y) / (2 ** quantizationLevel - 1) * np.log(1 + A)) - 1) / A
    return signs * expansion


def encodeMuLaw(x, quantizationLevel, mu=255):
    signs = np.sign(x)
    compression = np.log1p(mu * np.abs(x)) / np.log1p(mu)
    quantized = np.round(compression * (2 ** quantizationLevel - 1))
    return signs * quantized


def decodeMuLaw(y, quantizationLevel, mu=255):
    signs = np.sign(y)
    expansion = (np.expm1(np.abs(y) / (2 ** quantizationLevel - 1) * np.log1p(mu))) / mu
    return signs * expansion


def encodeDPCM(x, quantizationLevel):
    y = np.zeros(x.shape)
    e = 0
    for i in range(0, x.shape[0]):
        y[i] = quantize(x[i] - e, quantizationLevel)
        e += y[i]
    return y


def decodeDPCM():
    pass


document = Document()
document.add_heading('Laboratorium 07', 0)

singHigh, singHighFs = sf.read('sing_high2.wav', dtype='float32')
singMedium, singMediumFs = sf.read('sing_medium1.wav', dtype='float32')
singLow, singLowFs = sf.read('sing_low1.wav', dtype='float32')

# sings = [singHigh, singMedium, singLow]
# singsFs = [singHighFs, singMediumFs, singLowFs]
# for sing, singFs in zip(sings, singsFs):
#     originalSing = sing.copy()
#     plt.plot(np.arange(0, originalSing.shape[0]) / singFs, originalSing)
#     plt.show()
#
#     compressedALaw = encodeALaw(originalSing)
#     decompressedALaw = decodeALaw(compressedALaw)
#
#     plt.plot(np.arange(0, decompressedALaw.shape[0]) / singFs, decompressedALaw)
#     plt.show()

x = np.linspace(-1, 1, 1000)
y = 0.9 * np.sin(np.pi * x * 4)

plt.plot(x, y)
plt.xlim(-1, -0.75)
plt.ylim(0, 1)
plt.show()

yALaw = encodeALaw(y, 6)
yALawDecoded = decodeALaw(yALaw, 6)

plt.plot(x, yALawDecoded)
plt.xlim(-1, -0.75)
plt.ylim(0, 1)
plt.show()

yMuLaw = encodeMuLaw(y, 6)
yMuLawDecoded = decodeMuLaw(yMuLaw, 6)

plt.plot(x, yMuLawDecoded)
plt.xlim(-1, -0.75)
plt.ylim(0, 1)
plt.show()

yDPCM = encodeDPCM(y, 6)
yDPCMDecoded = decodeDPCM(y, 6)

plt.plot(x, yDPCMDecoded)
plt.xlim(-1, -0.75)
plt.ylim(0, 1)
plt.show()
