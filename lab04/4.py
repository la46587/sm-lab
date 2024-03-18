import numpy as np
import matplotlib.pyplot as plt


def imgToFloat(img):
    if np.issubdtype(img.dtype, np.floating):
        img = img[:, :, :3]
        return img
    else:
        img = img / 255.0
        return img


def generatePalette(N):
    pallete = np.linspace(0, 1, N).reshape(N, 1)
    return pallete


def colorFit(pixel, palette):
    closestColor = palette[np.argmin(np.linalg.norm(palette - pixel, axis=1))]
    return closestColor


def kwantColorFit(img, palette):
    outImg = img.copy()
    height = img.shape[0]
    width = img.shape[1]
    for w in range(height):
        for k in range(width):
            outImg[w, k] = colorFit(img[w, k], palette)
    return outImg

fig, axs = plt.subplots(2, 2)
fig.tight_layout()

img = plt.imread('GS_0001.tif')
img = imgToFloat(img)
axs[0, 0].imshow(img)
axs[0, 0].set_title('Original')

palette = generatePalette(2)
img2 = kwantColorFit(img, palette)
axs[0, 1].imshow(img2)
axs[0, 1].set_title('1-bit')

palette = generatePalette(4)
img4 = kwantColorFit(img, palette)
axs[1, 0].imshow(img4)
axs[1, 0].set_title('2-bit')

palette = generatePalette(16)
img16 = kwantColorFit(img, palette)
axs[1, 1].imshow(img16)
axs[1, 1].set_title('4-bit')
plt.show()
