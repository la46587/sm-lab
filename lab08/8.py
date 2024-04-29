from docx import Document
from docx.shared import Inches
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from scipy.fftpack import dct, idct


class JPEG:
    def __init__(self, Y, Cb, Cr, shape, Ratio, QY, QC):
        self.Y = Y
        self.Cb = Cb
        self.Cr = Cr
        self.shape = shape
        self.ChromaRatio = Ratio
        self.QY = QY
        self.QC = QC


def CompressJPEG(image, ratio="4:4:4", QY=None, QC=None):
    YCbCr = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    Y, Cb, Cr = YCbCr[:, :, 0], YCbCr[:, :, 1], YCbCr[:, :, 2]
    Y, Cb, Cr = subsample(Y, Cb, Cr)

    compressed_Y = compress_layer(Y, QY)
    compressed_Cb = compress_layer(Cb, QC)
    compressed_Cr = compress_layer(Cr, QC)

    return {
        'Y': compressed_Y, 'Cb': compressed_Cb, 'Cr': compressed_Cr,
        'shape': Y.shape, 'ChromaRatio': ratio, 'QY': QY, 'QC': QC
    }


def DecompressJPEG(jpeg):
    Y = decompress_layer(jpeg['Y'], jpeg['QY'], jpeg['shape'])
    Cb = decompress_layer(jpeg['Cb'], jpeg['QC'], jpeg['shape'])
    Cr = decompress_layer(jpeg['Cr'], jpeg['QC'], jpeg['shape'])

    decompressed_YCbCr = np.stack((Y, Cb, Cr), axis=-1)
    return cv2.cvtColor(decompressed_YCbCr.astype(np.uint8), cv2.COLOR_YCrCb2RGB)


def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')


def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')


def compress_block(block, Q):
    dct_block = dct2(block)
    quantized_block = np.round(dct_block / Q).astype(int)
    return quantized_block.flatten()


def decompress_block(vector, Q):
    reshaped_block = vector.reshape(8, 8)
    dequantized_block = reshaped_block * Q
    return idct2(dequantized_block)


def compress_layer(L, Q):
    S = np.array([])
    for w in range(0, L.shape[0], 8):
        for k in range(0, L.shape[1], 8):
            block = L[w:(w + 8), k:(k + 8)]
            S = np.append(S, compress_block(block, Q))
    return S


def decompress_layer(S, Q, dims):
    L = np.zeros(dims)
    num_blocks_w = dims[1] // 8
    for idx, i in enumerate(range(0, S.shape[0], 64)):
        vector = S[i:(i + 64)]
        w = (idx // num_blocks_w) * 8
        k = (idx % num_blocks_w) * 8
        L[w:(w + 8), k:(k + 8)] = decompress_block(vector, Q)
    return L


def subsample(Y, Cb, Cr):
    return Y, Cb, Cr


def encodeRLE(img):
    pixels = img.flatten()
    encoded = []
    count = 1
    prevPixel = pixels[0]

    for pixel in pixels[1:]:
        if np.array_equal(pixel, prevPixel):
            count += 1
        else:
            encoded.append((count, prevPixel))
            prevPixel = pixel
            count = 1
    encoded.append((count, prevPixel))
    return encoded


def calculateCompressionRatio(img, imgEncoded):
    originalSize = img.size
    encodedSize = len(imgEncoded)

    compressionRatio = abs(originalSize) / abs(encodedSize)
    compressionPercentage = 100 * (abs(encodedSize) / abs(originalSize))

    return compressionRatio, compressionPercentage


QY = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 36, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

QC = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])

document = Document()
document.add_heading('Laboratorium 08', 0)

images = ['BIG_0001.jpg', 'BIG_0003.jpg', 'moon.jpg', 'colored.jpg']

for image in images:
    memfile = BytesIO()

    input_image = mpimg.imread(image)
    compressed_image = CompressJPEG(input_image, "4:4:4", QY, QC)
    output_image = DecompressJPEG(compressed_image)

    imgEncodedRLE = encodeRLE(output_image.copy())
    compressionRatio, compressionPercentage = calculateCompressionRatio(output_image, imgEncodedRLE)

    fig, axs = plt.subplots(4, 2, sharey='all')
    fig.set_size_inches(9, 13)
    fig.tight_layout()

    axs[0, 0].imshow(input_image)
    axs[0, 0].axis('off')
    axs[0, 1].imshow(output_image)
    axs[0, 1].axis('off')

    input_ycbcr = cv2.cvtColor(input_image, cv2.COLOR_RGB2YCrCb)
    output_ycbcr = cv2.cvtColor(output_image, cv2.COLOR_RGB2YCrCb)

    for i in range(3):
        axs[i + 1, 0].imshow(input_ycbcr[:, :, i], cmap='gray')
        axs[i + 1, 0].axis('off')
        axs[i + 1, 1].imshow(output_ycbcr[:, :, i], cmap='gray')
        axs[i + 1, 1].axis('off')

    plt.savefig(memfile, bbox_inches='tight')
    document.add_heading(f'{image} - stopień kompresji: {compressionPercentage:.4f} ({compressionPercentage:.2f}%)', 2)
    document.add_picture(memfile, width=Inches(4.3))
    plt.close(fig)

document.save('lab08.docx')
