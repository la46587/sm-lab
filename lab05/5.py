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
    else:
        return print('invalid interpolation method')
    return interpolatedSignal


def plotAudio(Signal, Fs, timeMargin=[0, 0.1]):
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.xlim(timeMargin)
    plt.plot(np.arange(0, Signal.shape[0]) / Fs, Signal)
    plt.xlabel('czas trwania [s]')
    plt.subplot(2, 1, 2)
    yf = scipy.fftpack.fft(Signal, fsize)
    plt.plot(np.arange(0, fs / 2, fs / fsize), 20 * np.log10(np.abs(yf[:fsize // 2])))
    plt.xlabel('częstotliwość [Hz]')
    plt.ylabel('[dB]')
    plt.show()


np.seterr(divide='ignore')

data, fs = sf.read('sin_440Hz.wav', dtype='float32')
print(data.dtype)
print(data.shape)

print('int:', np.issubdtype(data.dtype, np.integer))
print('float:', np.issubdtype(data.dtype, np.floating))

x = np.arange(0, data.shape[0])

sd.play(data, fs)
status = sd.wait()

fsize = 2**16
plotAudio(data, fs)
