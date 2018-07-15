# this file is write for handling some image data
# the data is in TIFF format and i transform it into 4 single jpeg file
# with only ont single class on the image using ARCGIS and the ground truth image

import os
import random
import cv2
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array


class SingleClassImageHandler():
    def __init__(self, image_path, img_size):
        self.image_path = image_path
        self.img_size = img_size

    def load_image(self):
        base_image = cv2.imread(self.image_path, flags=cv2.IMREAD_UNCHANGED)
        image = base_image[:, :, 0:3]
        image = np.uint8(image)
        self.image = image
        return image

    def random_patch(self):
        row = self.image.shape[0]
        col = self.image.shape[1]
        r = random.randint(0, row - self.img_size)
        c = random.randint(0, col - self.img_size)
        sub_image = self.image[r:r + self.img_size, c:c + self.img_size]

        return sub_image

data_root = r'.\landuse'
commercial_image = r'commercial_clip_rgb.jpg'
Industrial_image = r'Industrial_clip_rgb.jpg'
Residentia_image = r'Residentia_clip_rgb.jpg'
Servicepublic_image = r'Servicepublic_clip_rgb.jpg'


img_size = 255