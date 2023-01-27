import os
import cv2
import torch
import numpy as np
from util import time_sync
from collections import Counter
from tools import load_model
from box import non_max_suppression


def pre_process(img, new_shape, stride, half, device):
    half = half & (device != 'cpu')

    # scale image
    shape = img.shape[:2]

    ratio = min(min(new_shape[0] / shape[0], new_shape[1] / shape[1]), 1)

    unpad_shape = (round(shape[1] * ratio), round(shape[0] * ratio))

    dW, dH = ((new_shape[1] - unpad_shape[0]) % stride) / 2, ((new_shape[0] - unpad_shape[1]) % stride) / 2

    if shape != unpad_shape:
        img = cv2.resize(img, unpad_shape, cv2.INTER_LINEAR)

    top, bottom = int(round(dH - 0.1)), int(round(dH + 0.1))
    left, right = int(round(dW - 0.1)), int(round(dW + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # pre-process
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = img / 255

    img = torch.from_numpy(img)
    img = img.to(torch.float16) if half else img.to(torch.float32)
    img = img.to(device)

    return img[None]


def rescale_box(shape0, shape1, box):
    gain = min(shape1[0] / shape0[0], shape1[1] / shape0[1])
    pad = (shape1[1] - shape0[1] * gain) / 2, (shape1[0] - shape0[0] * gain) / 2

    box[:, [0, 2]] -= pad[0]  # x padding
    box[:, [1, 3]] -= pad[1]  # y padding
    box[:, :4] /= gain

    box[:, 0].clamp_(0, shape0[1])
    box[:, 1].clamp_(0, shape0[0])
    box[:, 2].clamp_(0, shape0[1])
    box[:, 3].clamp_(0, shape0[0])

    return box


def save_detection(img, pred, cls, color, file):
    if len(pred) > 0:
        objs = [[k, v] for k, v in dict(Counter(pred[:, 5].tolist())).items()]
        objs.sort()
        info = ', '.join([str(x[1]) + ' ' + cls[int(x[0])] for x in objs]) + ', Done.'

        for line in pred:
            line = line.tolist()
            pt1 = (round(line[0]), round(line[1]))
            pt2 = (round(line[2]), round(line[3]))
            pt3 = (round(line[0]), round(line[1]) - 8)

            c = int(line[5])
            v = round(line[4], 2)
            cv2.rectangle(img, pt1, pt2, color[int(line[5])], 4)
            cv2.putText(img, cls[c] + ' ' + str(v), pt3, 0, 0.75, (255, 255, 255), 2)

    else:
        info = 'zero objects'

    cv2.imwrite(file, img)

    return info


def exec(cfg):
    training = False
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = load_model(cfg.name, cfg.anchors, cfg.strides, cfg.num_cls,
                       cfg.pretrain_path, cfg.cut_index, cfg.half, device, training)

    files = os.listdir(cfg.input_dir)
    num = len(files)
    c1, c2, c3, c4 = 0, 0, 0, 0
    with torch.no_grad():
        for i in range(num):
            file = files[i]

            img0 = cv2.imread(cfg.input_dir + '/' + file)

            t1 = time_sync()
            img1 = pre_process(img0, cfg.new_shape, cfg.stride, cfg.half, device)

            t2 = time_sync()
            pred = model(img1)

            t3 = time_sync()
            pred = non_max_suppression(pred, cfg.conf_t, cfg.multi_label, cfg.max_box,
                                       cfg.max_wh, cfg.iou_t, cfg.max_det, cfg.merge)

            t4 = time_sync()
            pred = pred[0]

            if len(pred) > 0:
                pred[:, :4] = rescale_box(img0.shape[:2], img1.shape[2:], pred[:, :4])

            t5 = time_sync()
            info = save_detection(img0, pred, cfg.cls, cfg.color, cfg.output_dir + '/' + file)

            c1 += t2 - t1
            c2 += t3 - t2
            c3 += t4 - t3
            c4 += t5 - t4

            print('image %d/%d' % (i + 1, num), file, info, f'({t5 - t1:.3})s')

        sen = 'Speed: %s pre-process, %s inference, %s NMS, %s rescale-box, %s per image'
        c1 = (c1 / num) * 1000
        c2 = (c2 / num) * 1000
        c3 = (c3 / num) * 1000
        c4 = (c4 / num) * 1000
        c5 = c1 + c2 + c3 + c4

        print(sen % (f'{(c1):.2f}ms', f'{(c2):.2f}ms', f'{(c3):.2f}ms', f'{(c4):.2f}ms', f'{(c5):.2f}ms'))
