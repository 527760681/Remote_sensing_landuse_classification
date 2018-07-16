# this file is write for handling some image data
# the data is in TIFF format and i transform it into 4 single jpeg file
# with only ont single class on the image using ARCGIS and the ground truth image

import os
import random
import cv2
import numpy as np

class SingleClassImageHandler():
    def __init__(self, image_path, img_size):
        self.image_path = image_path
        self.img_size = img_size

    def load_image(self):
        base_image = cv2.imread(self.image_path, flags=cv2.IMREAD_UNCHANGED)
        image = base_image[:, :, 0:3]
        image = np.uint8(image)
        return image


    def random_patch(self):
        image = self.load_image()
        row = image.shape[0]
        col = image.shape[1]
        r = random.randint(0, row - self.img_size)
        c = random.randint(0, col - self.img_size)
        sub_image = image[r:r + self.img_size, c:c + self.img_size]

        return sub_image

data_root = r'.\landuse'
sub_image_root = r'.\sub_image'
commercial_image = r'commercial_clip_rgb.jpg'
Industrial_image = r'Industrial_clip_rgb.jpg'
Residentia_image = r'Residentia_clip_rgb.jpg'
Servicepublic_image = r'Servicepublic_clip_rgb.jpg'

patch_num = 200
img_size = 256

image_content = [commercial_image,Industrial_image,Residentia_image,Servicepublic_image]

for image in image_content:
    handler = SingleClassImageHandler(os.path.join(data_root,image),img_size=img_size)

    class_name = image.split('.')[0]
    class_path = os.path.join(sub_image_root,class_name)
    if not os.path.exists(class_path):
        os.mkdir(class_path) #need to mkdir 'sub_image_root' handly

    for i in range(patch_num):
        sub_image = handler.random_patch()
        print('writing down the no.'+str(i)+' image in class '+image)
        cv2.imwrite(os.path.join(class_path,str(i)+'.jpg'),sub_image)
