from docx import Document
from docx.shared import Inches
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('technical.jpg')
plt.imshow(img)
plt.show()

img = img.astype(int)
ogShape = img.shape
img = img.flatten()
img = img.reshape(ogShape)
print(img.shape)
