import random
import cv2
import os
import numpy as np

img = cv2.imread('kot/kot.jpg')
path = 'C:/Users/insomnia/Desktop/sm-lab/lab10/kot'

for i in range(15):
    val1 = random.randint(5, 50)
    val2 = np.random.uniform(0, 150)
    val3 = np.random.uniform(0, 150)
    blur_b = cv2.bilateralFilter(img, val1, val2, val3)
    cv2.imwrite(os.path.join(path, str(i+1) + '.jpg'), blur_b)
