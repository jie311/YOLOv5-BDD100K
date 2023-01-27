import os
import json
import cv2
import numpy as np


def build_mask(mask_file):
    mask_image = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

    bit_image = np.vectorize(lambda i: '{0:08b}'.format(i))(mask_image)

    direction = np.vectorize(lambda i: i[2])(bit_image)
    style = np.vectorize(lambda i: i[3])(bit_image)
    backgroud = np.vectorize(lambda i: i[4])(bit_image)
    category = np.vectorize(lambda i: i[5:])(bit_image)

    return direction, style, backgroud, category


def build_sign(mask_file, type):
    direction, style, backgroud, category = build_mask(mask_file)

    sign = np.zeros([720, 1280, 3], dtype=np.uint8)

    if type == 'direction':
        sign[:, :, 0][direction == '0'] = 255  # parallel
        sign[:, :, 2][direction == '1'] = 255  # vertical

    if type == 'style':
        sign[:, :, 0][style == '0'] = 255  # solid
        sign[:, :, 2][style == '1'] = 255  # dashed

    if type == 'category':
        cls_colors = {
            '000': [0, 100, 255],  # crosswalk: orange
            '001': [0, 255, 0],  # double other: green
            '010': [255, 255, 255],  # double white: white
            '011': [0, 255, 255],  # double yellow: yellow
            '100': [0, 0, 255],  # road curb: red
            '101': [0, 255, 0],  # single other: green
            '110': [255, 255, 255],  # single white: white
            '111': [0, 255, 255]  # single yellow: yellow
        }

        for k, v in cls_colors.items():
            sign[:, :, :][category == k] = v

    sign[:, :, :][backgroud == '1'] = 0

    return sign


def show_lane(label_dir, image_dir, lane_dir, type='direction'):
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

                mask_file = lane_dir + '/' + sub_dir + '/' + name.replace('jpg', 'png')
                sign = build_sign(mask_file, type)

                image = cv2.addWeighted(image, 1, sign, 1, 1)

                cv2.namedWindow(name, 0)
                cv2.resizeWindow(name, 2560, 1440)
                cv2.imshow(name, image)
                cv2.waitKey(10 * 60 * 1000)
                cv2.destroyAllWindows()


label_dir = '../bdd100k/labels'
image_dir = '../bdd100k/images'
lane_mask_dir = '../bdd100k/lane/masks'

# type  direction, style, category
show_lane(label_dir, image_dir, lane_mask_dir, type='category')
