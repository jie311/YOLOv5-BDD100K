import cv2
import math
import torch
import numpy as np
from box import xywh_norm2xyxy
from collections import Counter
import matplotlib.pyplot as plt


def select_color(index):
    hex = ['#FF701F', '#FFB21D', '#CFD231', '#48F90A', '#92CC17', '#3DDB86', '#1A9334', '#00D4BB', '#FF3838', '#FF9D97',
           '#2C99A8', '#00C2FF', '#344593', '#6473FF', '#0018EC', '#8438FF', '#520085', '#CB38FF', '#FF95C8', '#FF37C7']

    c = hex[index % len(hex)]
    c = tuple(int(c[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    return (c[2], c[1], c[0])


def sign_lane(lane, lane_type):
    lane = np.apply_along_axis(lambda i: list(i).index(1), 2, lane)

    H, W = lane.shape
    sign = np.zeros([H, W, 3], dtype=np.uint8)

    if lane_type == 'direction':
        sign[:, :, 0][lane == 1] = 255  # parallel
        sign[:, :, 2][lane == 2] = 255  # vertical

    elif lane_type == 'style':
        sign[:, :, 0][lane == 1] = 255  # solid
        sign[:, :, 2][lane == 2] = 255  # dashed

    else:
        cls_colors = {
            1: [0, 100, 255],  # crosswalk: orange
            2: [0, 255, 0],  # double other: green
            3: [255, 255, 255],  # double white: white
            4: [0, 255, 255],  # double yellow: yellow
            5: [0, 0, 255],  # road curb: red
            6: [0, 255, 0],  # single other: green
            7: [255, 255, 255],  # single white: white
            8: [0, 255, 255]  # single yellow: yellow
        }

        for k, v in cls_colors.items():
            sign[:, :, :][lane == k] = v

    return sign


def sign_drivable(drivable):
    drivable = np.apply_along_axis(lambda i: list(i).index(1), 2, drivable)
    H, W = drivable.shape
    sign = np.zeros([H, W, 3], dtype=np.uint8)
    sign[:, :, :][drivable == 1] = [70, 55, 20]  # direct
    sign[:, :, :][drivable == 2] = [50, 25, 50]  # alternative

    return sign


def plot_images(imgs, lanes, drivables, lane_type, labels, save_file, max_shape=(2880, 3840), max_subplots=16):
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.cpu().float().numpy()

    if (imgs[0].shape[1:3] == lanes[0].shape[0:2]) and isinstance(lanes, torch.Tensor):
        lanes = lanes.cpu().float().numpy()

    if (imgs[0].shape[1:3] == drivables[0].shape[0:2]) and isinstance(drivables, torch.Tensor):
        drivables = drivables.cpu().float().numpy()

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().float().numpy()

    if np.max(imgs[0]) <= 1:
        imgs *= 255.0

    B, C, H, W = imgs.shape
    B = min(B, max_subplots)
    num = np.ceil(B ** 0.5)

    labels[:, 2:6] = xywh_norm2xyxy(labels[:, 2:6], W, H, 0, 0)

    img_mosaic = np.full((int(num * H), int(num * W), 3), 0, dtype=np.uint8)
    lane_mosaic = np.full((int(num * H), int(num * W), 3), 0, dtype=np.uint8)
    drivable_mosaic = np.full((int(num * H), int(num * W), 3), 0, dtype=np.uint8)

    for index, img in enumerate(imgs):
        if index == max_subplots:
            break

        x, y = int(W * (index // num)), int(H * (index % num))
        img = img[::-1].transpose(1, 2, 0)
        img = np.ascontiguousarray(img)
        img_mosaic[y:y + H, x:x + W, :] = img
        cv2.rectangle(img_mosaic, (x, y), (x + W, y + H), (255, 255, 255), 2)

        if imgs[0].shape[1:3] == lanes[0].shape[0:2]:
            lane = sign_lane(lanes[index], lane_type)
            lane_mosaic[y:y + H, x:x + W, :] = lane

        if imgs[0].shape[1:3] == drivables[0].shape[0:2]:
            drivable = sign_drivable(drivables[index])
            drivable_mosaic[y:y + H, x:x + W, :] = drivable

        img_labels = labels[labels[:, 0] == index]
        for img_label in img_labels:
            color = select_color(int(img_label[1]))
            pt1 = (round(img_label[2]) + x, round(img_label[3]) + y)
            pt2 = (round(img_label[4]) + x, round(img_label[5]) + y)
            cv2.rectangle(img_mosaic, pt1, pt2, color, 2)

    ratio = min(max_shape[0] / num / H, max_shape[1] / num / W)
    if ratio < 1:
        H = math.ceil(ratio * H)
        W = math.ceil(ratio * W)
        img_mosaic = cv2.resize(img_mosaic, tuple(int(x * num) for x in (W, H)), interpolation=cv2.INTER_LINEAR)
        lane_mosaic = cv2.resize(lane_mosaic, tuple(int(x * num) for x in (W, H)), interpolation=cv2.INTER_NEAREST)
        drivable_mosaic = cv2.resize(drivable_mosaic, tuple(int(x * num) for x in (W, H)),
                                     interpolation=cv2.INTER_NEAREST)

    if len(lanes):
        img_mosaic = cv2.addWeighted(img_mosaic, 1, lane_mosaic, 1, 1)

    if len(drivables):
        img_mosaic = cv2.addWeighted(img_mosaic, 1, drivable_mosaic, 1, 1)

    cv2.imwrite(save_file, img_mosaic)


def plot_labels(labels, targets, save_file):
    indices = []
    centers = []
    whs = []

    for label in labels:
        indices.extend([int(x) for x in label[:, 0].tolist()])
        centers.extend(label[:, 1:3].tolist())
        whs.extend(label[:, 3:5].tolist())

    count = dict(Counter(indices))
    t_indices = sorted(count.items(), key=lambda x: x[0])
    t_data = [[targets[x[0]], x[1]] for x in t_indices]

    plt.figure(figsize=(15, 9))
    plt.subplot(2, 1, 1)
    plt.title('target', fontsize=12)
    for k, v in t_data:
        bar = plt.bar(k, v, log=True)
        plt.bar_label(bar, fontsize=8)

    plt.subplot(2, 2, 3)
    plt.title('center', fontsize=12)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter([x[0] for x in centers], [x[1] for x in centers], s=0.05)

    plt.subplot(2, 2, 4)
    plt.title('wh', fontsize=12)
    plt.xlabel('w')
    plt.ylabel('h')
    plt.scatter([x[0] for x in whs], [x[1] for x in whs], s=0.05)

    plt.savefig(save_file)
