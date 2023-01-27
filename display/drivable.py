import os
import json
import cv2
import numpy as np


def build_sign(mask_file):
    mask_image = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

    sign = np.zeros([720, 1280, 3], dtype=np.uint8)

    sign[:, :, :][mask_image == 0] = [70, 55, 20]  # direct
    sign[:, :, :][mask_image == 1] = [50, 25, 50]  # alternative

    return sign


def show_drivable(label_dir, image_dir, drivable_mask_dir):
    files = os.listdir(label_dir)
    files.sort()

    for file in files:
        with open(label_dir + '/' + file, 'r', encoding='utf-8') as f:
            lines = json.load(f)

            print('reading {}, {}'.format(file, len(lines)))

            sub_dir = file.split('_')[-1].replace('.json', '')
            for line in lines:
                name = line['name']
                print(name)

                image = cv2.imread(image_dir + '/' + sub_dir + '/' + name)

                mask_file = drivable_mask_dir + '/' + sub_dir + '/' + name.replace('jpg', 'png')
                sign = build_sign(mask_file)

                image = cv2.addWeighted(image, 1, sign, 1, 1)

                cv2.namedWindow(name, 0)
                cv2.resizeWindow(name, 2560, 1440)
                cv2.imshow(name, image)
                cv2.waitKey(10 * 60 * 1000)
                cv2.destroyAllWindows()


label_dir = '../bdd100k/labels'
image_dir = '../bdd100k/images'
drivable_mask_dir = '../bdd100k/drivable/masks'

show_drivable(label_dir, image_dir, drivable_mask_dir)
