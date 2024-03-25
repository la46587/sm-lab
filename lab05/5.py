from docx import Document
from docx.shared import Inches
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import sounddevice as sd
import soundfile as sf


def quantize(signal, targetBits):
    sourceVal = 2 ** 32 - 1
    targetVal = 2 ** targetBits - 1

    scaledSignal = [(sample * targetVal / sourceVal) for sample in signal]
    quantizedSignal = [round(sample) for sample in scaledSignal]
    return quantizedSignal


def decimate(signal, n):
    decimatedSignal = signal[::n]
    return decimatedSignal


def interpolate(signal, rate, method):
    oldSignal = np.arange(0, len(signal))
    newSignal = np.linspace(0, len(signal) - 1, rate)

    if method == 'linear':
        interpolatedSignal = np.interp(newSignal, oldSignal, signal)
    elif method == 'cubic':
        interpolatedSignal = interpolate.interp1d(oldSignal, signal, kind='cubic')
    elif method == 'lagrange':
        poly = interpolate.lagrange(oldSignal, signal)
        interpolatedSignal = np.vectorize(lambda x: poly(x))
    return interpolatedSignal


data, fs = sf.read('sing_high1.wav', dtype='float32')
print(data.dtype)
print(data.shape)

print(np.issubdtype(data.dtype, np.integer))
print(np.issubdtype(data.dtype, np.floating))

x = np.arange(0, data.shape[0])
plt.plot(x, data)
plt.show()
