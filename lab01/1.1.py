import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import sounddevice as sd
import soundfile as sf

# Sygnał w czasie

data, fs = sf.read('sound1.wav', dtype='float32')
print(data.dtype)
print(data.shape)

# sd.play(data, fs)
# status = sd.wait()

data_L = data[:, 0]
data_R = data[:, 1]
data_mix = (np.array(data_L) + np.array(data_R)) / 2

sf.write('sound_L.wav', data_L, fs)
sf.write('sound_R.wav', data_R, fs)
sf.write('sound_mix.wav', data_mix, fs)

x = np.arange(0, data.shape[0])
x = x / fs

plt.subplot(2, 1, 1)
plt.plot(x, data_L)
plt.subplot(2, 1, 2)
plt.plot(x, data_R)
plt.show()

# Widmo dźwięku

data, fs = sf.read('sin_440Hz.wav', dtype=np.int32)
fsize = 2**8

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(np.arange(0, data.shape[0]) / fs, data)
plt.subplot(2, 1, 2)
yf = scipy.fftpack.fft(data, fsize)
plt.plot(np.arange(0, fs / 2, fs / fsize), 20 * np.log10(np.abs(yf[:fsize // 2])))
plt.show()
