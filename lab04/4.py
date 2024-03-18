import numpy as np
import matplotlib.pyplot as plt
import cv2


def imgToFloat(img):
    if np.issubdtype(img.dtype, np.floating):
        img = img[:, :, :3]
        return img
    else:
        img = img / 255.0
        img = img[:, :, :3]
        return img


def generatePalette(N):
    pallete = np.linspace(0, 1, N).reshape(N, 1)
    return pallete


def colorFit(pixel, palette):
    closestColor = palette[np.argmin(np.linalg.norm(palette - pixel, axis=1))]
    return closestColor


def kwantColorFit(img, palette):
    outImg = img.copy()
    for w in range(img.shape[0]):
        for k in range(img.shape[1]):
            outImg[w, k] = colorFit(img[w, k], palette)
    return outImg


def ditheringRandom(img):
    r = np.random.rand(*img.shape)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j][0] >= r[i, j][0]:
                img[i, j] = 1
            else:
                img[i, j] = 0
    return img


def ditheringOrganized(img):
    pass


def ditheringFloydSteinberg(img):
    pass


palette8 = np.array([
        [0.0, 0.0, 0.0,],
        [0.0, 0.0, 1.0,],
        [0.0, 1.0, 0.0,],
        [0.0, 1.0, 1.0,],
        [1.0, 0.0, 0.0,],
        [1.0, 0.0, 1.0,],
        [1.0, 1.0, 0.0,],
        [1.0, 1.0, 1.0,],
])

palette16 = np.array([
        [0.0, 0.0, 0.0,],
        [0.0, 1.0, 1.0,],
        [0.0, 0.0, 1.0,],
        [1.0, 0.0, 1.0,],
        [0.0, 0.5, 0.0,],
        [0.5, 0.5, 0.5,],
        [0.0, 1.0, 0.0,],
        [0.5, 0.0, 0.0,],
        [0.0, 0.0, 0.5,],
        [0.5, 0.5, 0.0,],
        [0.5, 0.0, 0.5,],
        [1.0, 0.0, 0.0,],
        [0.75, 0.75, 0.75,],
        [0.0, 0.5, 0.5,],
        [1.0, 1.0, 1.0,],
        [1.0, 1.0, 0.0,]
])

# fig, axs = plt.subplots(2, 2)
# fig.tight_layout()
#
# img = cv2.imread('GS_0002.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = imgToFloat(img)
#
# axs[0, 0].imshow(img)
# axs[0, 0].set_title('Original')
#
# palette = generatePalette(2)
# img1 = kwantColorFit(img, palette)
# axs[0, 1].imshow(img1)
# axs[0, 1].set_title('1-bit')
#
# palette = generatePalette(4)
# img2 = kwantColorFit(img, palette)
# axs[1, 0].imshow(img2)
# axs[1, 0].set_title('2-bit')
#
# palette = generatePalette(16)
# img4 = kwantColorFit(img, palette)
# axs[1, 1].imshow(img4)
# axs[1, 1].set_title('4-bit')
# plt.show()
#
# plt.close(fig)
# img8 = kwantColorFit(img, palette8)
# plt.imshow(img8)
# plt.show()
#
# img16 = kwantColorFit(img, palette16)
# plt.imshow(img16)
# plt.show()

img = cv2.imread('GS_0002.png')
img = imgToFloat(img)
plt.imshow(img)
plt.show()

img1 = kwantColorFit(img, generatePalette(2))
plt.imshow(img1)
plt.show()

imgDitherRandom = ditheringRandom(img)
plt.imshow(imgDitherRandom)
plt.show()

imgDitherOrganized = ditheringOrganized(img)
plt.imshow(imgDitherOrganized)
plt.show()
