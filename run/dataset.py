import cv2
import json
import torch
import random
import imagesize
import numpy as np
from tqdm import tqdm
from util import to_one_hot
from torch.utils.data import Dataset
from box import xywh_norm2xyxy, xyxy2xywh_norm


def build_labels(input_file, output_file, image_dir, cfg):
    yolo_labels = []

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = json.load(f)

        for index in tqdm(range(len(lines)), desc=f'reading {input_file}, {len(lines)} records'):
            name = lines[index]['name']
            labels = lines[index]['labels']
            W, H = imagesize.get(image_dir + '/' + name)

            targets = []
            for label in labels:
                category = label['category']

                if category in cfg:
                    cls = cfg.index(category)

                    box2d = label['box2d']
                    x1, y1, x2, y2 = box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']
                    x = round((x1 + x2) / (2 * W), 4)
                    y = round((y1 + y2) / (2 * H), 4)
                    w = round(abs(x1 - x2) / W, 4)
                    h = round(abs(y1 - y2) / H, 4)

                    targets.append(','.join([str(i) for i in [cls, x, y, w, h]]))

            yolo_labels.append(name + '  ' + str(W) + ',' + str(H) + '  ' + '  '.join(targets))

    with open(output_file, 'w', encoding='utf-8') as f:
        for label in yolo_labels:
            f.write(label + '\n')


def read_labels(input_file):
    imgs, labels, indices, shapes = [], [], [], []
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

        print('reading {}, {} coco'.format(input_file, len(lines)))
        for index in range(len(lines)):
            indices.append(index)

            line = lines[index].replace('\n', '').split("  ")

            imgs.append(line[0])
            shapes.append([int(x) for x in line[1].split(',')])
            labels.append(np.array([[float(i) for i in obj.split(',')] for obj in line[2:]]))

    return np.array(imgs), np.array(labels), np.array(indices), np.array(shapes)


def read_lane(lane_file, lane_type):
    lane = cv2.imread(lane_file, cv2.IMREAD_GRAYSCALE)

    bit_image = np.vectorize(lambda i: '{0:08b}'.format(i))(lane)
    backgroud = np.vectorize(lambda i: i[4])(bit_image)

    if lane_type == 'direction':
        direction = np.vectorize(lambda i: i[2])(bit_image)
        direction = direction.astype(np.uint8)
        direction += 1
        direction[:][backgroud == '1'] = 0

        return direction

    elif lane_type == 'style':
        style = np.vectorize(lambda i: i[3])(bit_image)
        style = style.astype(np.uint8)
        style += 1
        style[:][backgroud == '1'] = 0

        return style

    else:
        category = np.vectorize(lambda i: i[5:])(bit_image)
        category = np.vectorize(lambda i: int(i[0]) * 4 + int(i[1]) * 2 + int(i[2]))(category)
        category = category.astype(np.uint8)
        category += 1
        category[:][backgroud == '1'] = 0

        return category


def read_drivable(drivable_file):
    drivable = cv2.imread(drivable_file, cv2.IMREAD_GRAYSCALE)
    drivable += 1
    drivable[:][drivable == 3] = 0

    return drivable


def load_image(img_file, lane_file, drivable_file, lane_type, size):
    img = cv2.imread(img_file)

    lane, drivable = np.array([]), np.array([])
    if lane_file != '':
        lane = read_lane(lane_file, lane_type)

    if drivable_file != '':
        drivable = read_drivable(drivable_file)

    shape = img.shape[:2]
    ratio = size / max(shape[0], shape[1])
    if ratio != 1:
        img = cv2.resize(img, (int(shape[1] * ratio), int(shape[0] * ratio)), interpolation=cv2.INTER_LINEAR)

        if len(lane):
            lane = cv2.resize(lane, (int(shape[1] * ratio), int(shape[0] * ratio)), interpolation=cv2.INTER_NEAREST)

        if len(drivable):
            drivable = cv2.resize(drivable, (int(shape[1] * ratio), int(shape[0] * ratio)),
                                  interpolation=cv2.INTER_NEAREST)

    return img, lane, drivable, img.shape[:2]


def letterbox(img, lane, drivable, size, stride):
    shape = img.shape[:2]
    ratio = min(size / max(shape[0], shape[1]), 1)
    unpad_shape = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))
    pad_w, pad_h = ((size - unpad_shape[0]) % stride) / 2, ((size - unpad_shape[1]) % stride) / 2

    if shape[::-1] != unpad_shape:
        img = cv2.resize(img, unpad_shape, interpolation=cv2.INTER_LINEAR)

        if len(lane):
            lane = cv2.resize(lane, unpad_shape, interpolation=cv2.INTER_NEAREST)

        if len(drivable):
            drivable = cv2.resize(drivable, unpad_shape, interpolation=cv2.INTER_NEAREST)

    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    if len(lane):
        lane = cv2.copyMakeBorder(lane, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0))

    if len(drivable):
        drivable = cv2.copyMakeBorder(drivable, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0))

    return img, lane, drivable, ratio, (pad_w, pad_h)


def mosaic(img_dir, lane_dir, drivable_dir, lane_type, imgs, labels, indices, index, new_shape):
    c_x = int(random.uniform(new_shape[1] // 2, 2 * new_shape[1] - new_shape[1] // 2))
    c_y = int(random.uniform(new_shape[0] // 2, 2 * new_shape[0] - new_shape[0] // 2))

    t_indices = [index] + random.choices(indices, k=3)
    random.shuffle(t_indices)

    img4 = np.full((new_shape[0] * 2, new_shape[1] * 2, 3), 114, dtype=np.uint8)
    img4_labels = []

    img4_lanes, img4_drivables = np.array([]), np.array([])
    if lane_dir != '':
        img4_lanes = np.full((new_shape[0] * 2, new_shape[1] * 2), 0, dtype=np.uint8)

    if drivable_dir != '':
        img4_drivables = np.full((new_shape[0] * 2, new_shape[1] * 2), 0, dtype=np.uint8)

    for (i, index) in enumerate(t_indices):
        img_name = imgs[index]
        img_file = img_dir + '/' + img_name
        lane_file = lane_dir + '/' + img_name.replace('jpg', 'png') if lane_dir != '' else ''
        drivable_file = drivable_dir + '/' + img_name.replace('jpg', 'png') if drivable_dir != '' else ''

        img, lane, drivable, (h, w) = load_image(img_file, lane_file, drivable_file, lane_type, max(new_shape))

        if i == 0:
            x1a, y1a, x2a, y2a = max(c_x - w, 0), max(c_y - h, 0), c_x, c_y
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:
            x1a, y1a, x2a, y2a = c_x, max(c_y - h, 0), min(c_x + w, new_shape[1] * 2), c_y
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:
            x1a, y1a, x2a, y2a = max(c_x - w, 0), c_y, c_x, min(new_shape[0] * 2, c_y + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:
            x1a, y1a, x2a, y2a = c_x, c_y, min(c_x + w, new_shape[1] * 2), min(new_shape[0] * 2, c_y + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]

        if len(img4_lanes):
            img4_lanes[y1a:y2a, x1a:x2a] = lane[y1b:y2b, x1b:x2b]

        if len(img4_drivables):
            img4_drivables[y1a:y2a, x1a:x2a] = drivable[y1b:y2b, x1b:x2b]

        # labels
        offset_x, offset_y = x1a - x1b, y1a - y1b
        img_labels = labels[index].copy()

        if len(img_labels):
            img_labels[:, 1:5] = xywh_norm2xyxy(img_labels[:, 1:5], w, h, offset_x, offset_y)
            img4_labels.append(img_labels)

    img4_labels = np.concatenate(img4_labels, 0)

    img4_labels[:, [1, 3]] = np.clip(img4_labels[:, [1, 3]], 0, 2 * new_shape[1])
    img4_labels[:, [2, 4]] = np.clip(img4_labels[:, [2, 4]], 0, 2 * new_shape[0])

    img4 = cv2.resize(img4, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_LINEAR)

    if len(img4_lanes):
        img4_lanes = cv2.resize(img4_lanes, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_NEAREST)

    if len(img4_drivables):
        img4_drivables = cv2.resize(img4_drivables, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_NEAREST)

    img4_labels = img4_labels / 2

    return img4, img4_lanes, img4_drivables, img4_labels


def affine_transform(img, lane, drivable, labels, scale, translate):
    H, W = img.shape[:2]
    # center
    C = np.eye(3)
    C[0, 2] = -W / 2
    C[1, 2] = -H / 2

    # scale
    R = np.eye(3)
    scale = random.uniform(1.25 - scale, 1.25 + scale)
    R[0, 0], R[1, 1] = scale, scale

    # translate
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * W
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * H

    M = T @ R @ C

    img = cv2.warpAffine(img, M[:2], dsize=(W, H), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    if len(lane):
        lane = cv2.warpAffine(lane, M[:2], dsize=(W, H), flags=cv2.INTER_NEAREST, borderValue=(0))

    if len(drivable):
        drivable = cv2.warpAffine(drivable, M[:2], dsize=(W, H), flags=cv2.INTER_NEAREST, borderValue=(0))

    num = len(labels)
    if num:
        points = np.ones((num * 4, 3))
        points[:, :2] = labels[:, 1:5][:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(num * 4, 2)
        points = points @ M.T
        points = points[:, :2].reshape(num, 8)

        x = points[:, [0, 2, 4, 6]]
        y = points[:, [1, 3, 5, 7]]
        box = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, num).T
        box[:, [0, 2]] = box[:, [0, 2]].clip(0, W)
        box[:, [1, 3]] = box[:, [1, 3]].clip(0, H)

        labels[:, 1:5] = box

    return img, lane, drivable, labels


def augment_hsv(img, h=0.015, s=0.7, v=0.4):
    if h or s or v:
        ratio = np.random.uniform(-1, 1, 3) * [h, s, v] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype

        x = np.arange(0, 256, dtype=ratio.dtype)
        lut_hue = ((x * ratio[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * ratio[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * ratio[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)


def flip_up_down(img, lane, drivable, labels):
    img = np.flipud(img)

    if len(lane):
        lane = np.flipud(lane)

    if len(drivable):
        drivable = np.flipud(drivable)

    if len(labels):
        labels[:, 2] = 1 - labels[:, 2]

    return img, lane, drivable, labels


def flip_left_right(img, lane, drivable, labels):
    img = np.fliplr(img)

    if len(lane):
        lane = np.fliplr(lane)

    if len(drivable):
        drivable = np.fliplr(drivable)

    if len(labels):
        labels[:, 1] = 1 - labels[:, 1]

    return img, lane, drivable, labels


def check_labels(labels, box_t, wh_rt, eps=1e-3):
    if len(labels):
        w, h = labels[:, 3] - labels[:, 1], labels[:, 4] - labels[:, 2]
        wh_r = np.maximum(w / (h + eps), h / (w + eps))
        index = (w > box_t) & (h > box_t) & (wh_r < wh_rt)
        labels = labels[index]

    return labels


class LoadDataset(Dataset):
    def __init__(self, img_dir, lane_dir, drivable_dir, lane_type, label_file, cfg):
        self.img_dir = img_dir
        self.lane_dir = lane_dir
        self.drivable_dir = drivable_dir
        self.lane_type = lane_type if lane_type != '' else 'category'
        self.cfg = cfg

        self.imgs, self.labels, self.indices, self.shapes = read_labels(label_file)

    def __getitem__(self, index):
        if self.cfg['augment'] and self.cfg['mosaic']:
            img, lane, drivable, labels = mosaic(self.img_dir, self.lane_dir, self.drivable_dir, self.lane_type,
                                                 self.imgs, self.labels, self.indices, index, self.cfg['shape'])

        else:
            img_name = self.imgs[index]
            img_file = self.img_dir + '/' + img_name
            lane_file = self.lane_dir + '/' + img_name.replace('jpg', 'png') if self.lane_dir != '' else ''
            drivable_file = self.drivable_dir + '/' + img_name.replace('jpg', 'png') if self.drivable_dir != '' else ''

            img, lane, drivable, (h, w) = load_image(img_file, lane_file, drivable_file,
                                                     self.lane_type, max(self.cfg['shape']))

            img, lane, drivable, ratio, (pad_w, pad_h) = letterbox(img, lane, drivable,
                                                                   max(self.cfg['shape']), self.cfg['strides'][-1])

            labels = self.labels[index].copy()
            if len(labels):
                labels[:, 1:5] = xywh_norm2xyxy(labels[:, 1:5], ratio * w, ratio * h, pad_w, pad_h)

        if self.cfg['augment'] and self.cfg['affine']:
            img, lane, drivable, labels = affine_transform(img, lane, drivable, labels,
                                                           self.cfg['scale'], self.cfg['translate'])

        labels = check_labels(labels, self.cfg['box_t'], self.cfg['wh_rt'])

        num = len(labels)
        if num:
            labels[:, 1:5] = xyxy2xywh_norm(labels[:, 1:5], img.shape[1], img.shape[0])

        if self.cfg['augment'] and random.random() < self.cfg['hsv']:
            augment_hsv(img)

        if self.cfg['augment'] and random.random() < self.cfg['flipud']:
            img, lane, drivable, labels = flip_up_down(img, lane, drivable, labels)

        if self.cfg['augment'] and random.random() < self.cfg['fliplr']:
            img, lane, drivable, labels = flip_left_right(img, lane, drivable, labels)

        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)

        lane = to_one_hot(lane, len(self.cfg['lane'][self.lane_type])) if len(lane) else torch.zeros(1)
        lane = torch.from_numpy(lane)

        drivable = to_one_hot(drivable, len(self.cfg['drivable'])) if len(drivable) else torch.zeros(1)
        drivable = torch.from_numpy(drivable)

        labels = torch.from_numpy(np.insert(labels, obj=0, values=0, axis=1)) if num else torch.zeros((num, 6))

        return img, lane, drivable, labels

    @staticmethod
    def collate_fn(batch):
        img, lane, drivable, labels = zip(*batch)

        for index, label in enumerate(labels):
            label[:, 0] = index

        return torch.stack(img, 0), torch.stack(lane, 0), torch.stack(drivable, 0), torch.cat(labels, 0)

    def __len__(self):
        return len(self.imgs)
