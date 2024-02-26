import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import sounddevice as sd
import soundfile as sf


def plotAudio(Signal, Fs, TimeMargin=[0, 0.02]):
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.xlim(TimeMargin)
    plt.plot(np.arange(0, data.shape[0]) / Fs, Signal)
    plt.xlabel('czas trwania [s]')
    plt.subplot(2, 1, 2)
    yf = scipy.fftpack.fft(Signal, fsize)
    plt.plot(np.arange(0, fs / 2, fs / fsize), 20 * np.log10(np.abs(yf[:fsize // 2])))
    plt.xlabel('częstotliwość [Hz]')
    plt.ylabel('[dB]')
    plt.show()


data, fs = sf.read('sin_60Hz.wav', dtype=np.int32)

fsize = 2**8
plotAudio(data, fs)
