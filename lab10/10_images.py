import random
import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def add_random_noise(image, intensity):
    noisy_image = image.copy()
    noise = np.random.randint(-intensity, intensity + 1, noisy_image.shape)
    noisy_image = np.clip(noisy_image + noise, 0, 255).astype(np.uint8)
    return noisy_image


img = cv2.imread('kot/kot.jpg')
path = 'C:/Users/insomnia/Desktop/sm-lab/lab10/kot'

for i in range(15):
    val1 = random.randint(5, 50)
    val2 = np.random.uniform(0, 150)
    val3 = np.random.uniform(0, 150)
    blur = cv2.bilateralFilter(img, val1, val2, val3)
    cv2.imwrite(os.path.join(path, str(i + 1) + '.jpg'), blur)

img = cv2.imread('mazda/mazda.jpg')
path = 'C:/Users/insomnia/Desktop/sm-lab/lab10/mazda'

for i in range(15):
    noise = add_random_noise(img, random.randint(1, 200))
    cv2.imwrite(os.path.join(path, str(i + 1) + '.jpg'), noise)

img = cv2.imread('zima/zima.png')
path = 'C:/Users/insomnia/Desktop/sm-lab/lab10/zima'

for i in range(15):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(5, 65)]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    cv2.imwrite(os.path.join(path, str(i + 1) + '.jpg'), decimg)
