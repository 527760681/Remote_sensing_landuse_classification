# this file is write for handling some image data
# the data is in TIFF format and i transform it into 4 single jpeg file
# with only ont single class on the image using ARCGIS and the ground truth image
import csv
import os
import random
import cv2
import numpy as np
from PIL import Image
from keras.utils import to_categorical
from sklearn.cross_validation import train_test_split


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

    def isBlank(self, sub_image):
        '''
        return true when the image has a blank pixel
        :param sub_image:
        :return:
        '''
        row = sub_image.shape[0]
        col = sub_image.shape[1]
        for r in range(row):
            for c in range(col):
                (b, g, red) = sub_image[r, c]  # read the bgr value in every point of the sub image
                if (b, g, red) == (0, 0, 0):
                    return True

        return False


# data_root = r'.\landuse'
# sub_image_root = r'.\sub_image'
# csv_path = r'class.txt'
# commercial_image = r'commercial_clip_rgb.jpg'
# Industrial_image = r'Industrial_clip_rgb.jpg'
# Residentia_image = r'Residentia_clip_rgb.jpg'
# Servicepublic_image = r'Servicepublic_clip_rgb.jpg'

data_root = r'.\testRegion'
sub_image_root = r'.\testRegion\sub_image'
csv_path = r'.\testRegion\class.txt'
commercial_image = r'commercial_clip_rgb.jpg'
Industrial_image = r'Industrial_clip_rgb.jpg'
Residentia_image = r'Residentia_clip_rgb.jpg'
Servicepublic_image = r'Servicepublic_clip_rgb.jpg'

patch_num = 100
img_size = 128
num_classes = 4



# '''
# split image into target size
# '''
# image_content = [commercial_image, Industrial_image, Residentia_image, Servicepublic_image]
#
# for image in image_content:
#     handler = SingleClassImageHandler(os.path.join(data_root, image), img_size=img_size)
#
#     class_name = image.split('.')[0]
#     class_path = os.path.join(sub_image_root, class_name)
#     if not os.path.exists(class_path):
#         os.mkdir(class_path)  # need to mkdir 'sub_image_root' handly
#
#     count = 0
#     while count < patch_num:
#         sub_image = handler.random_patch()
#         if not handler.isBlank(sub_image):
#             print('writing down the no.' + str(count) + ' image in class ' + image)
#             cv2.imwrite(os.path.join(class_path, str(count) + '.jpg'), sub_image)
#             count += 1

def write_csv(sub_image_root, csv_path):
    floders = os.listdir(sub_image_root)

    f = open(csv_path, 'a', newline='', encoding='utf8')
    writer = csv.writer(f)

    class_num = 0

    for floder in floders:
        floder = os.path.join(sub_image_root, floder)
        files = os.listdir(floder)
        for file in files:
            abs_path = os.path.join(floder, file)
            print('writing down the ' + file + ' to ' + csv_path)
            writer.writerow([abs_path, class_num])
        class_num += 1

    f.close()

# write_csv(sub_image_root,csv_path)


def load_data(csv_path,num_classes):
    X = []
    Y = []
    reader = csv.reader(open(csv_path, 'r', encoding='utf8'))
    for img_path, class_num in reader:
        img = Image.open(img_path)
        img_array = np.array(img)
        X.append(img_array)
        Y.append(class_num)

    data = np.stack(X, axis=0)
    label = np.stack(Y, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=42)
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    return X_train, X_test, y_train, y_test

def load_test_data(csv_path,num_classes):
    X = []
    Y = []
    reader = csv.reader(open(csv_path, 'r', encoding='utf8'))
    for img_path, class_num in reader:
        img = Image.open(img_path)
        img_array = np.array(img)
        X.append(img_array)
        Y.append(class_num)

    data = np.stack(X, axis=0)
    label = np.stack(Y, axis=0)

    # X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=42)
    X_test =data
    y_test = to_categorical(label, num_classes=num_classes)

    return X_test, y_test