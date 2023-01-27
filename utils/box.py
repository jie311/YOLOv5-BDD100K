import math
import torch
import torchvision
import numpy as np


def xywh2xyxy(box):
    if isinstance(box, torch.Tensor):
        x, y, w, h = box.chunk(4, 1)

        return torch.cat((x - w / 2, y - h / 2, x + w / 2, y + h / 2), 1)

    else:
        x, y, w, h = box[:, 0], box[:, 1], box[:, 2], box[:, 3]

        return np.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2), 1)


def xyxy2xywh(box):
    if isinstance(box, torch.Tensor):
        x1, y1, x2, y2 = box.chunk(4, 1)

        return torch.cat(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), 1)

    else:
        x1, y1, x2, y2 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]

        return np.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), 1)


def xywh_norm2xyxy(box, w, h, offset_x, offset_y):
    if isinstance(box, torch.Tensor):
        bx, by, bw, bh = box.chunk(4, 1)

        return torch.cat((w * (bx - bw / 2) + offset_x,
                          h * (by - bh / 2) + offset_y,
                          w * (bx + bw / 2) + offset_x,
                          h * (by + bh / 2) + offset_y), 1)

    else:
        bx, by, bw, bh = box[:, 0], box[:, 1], box[:, 2], box[:, 3]

        return np.stack((w * (bx - bw / 2) + offset_x,
                         h * (by - bh / 2) + offset_y,
                         w * (bx + bw / 2) + offset_x,
                         h * (by + bh / 2) + offset_y), 1)


def xyxy2xywh_norm(box, w, h):
    if isinstance(box, torch.Tensor):
        x1, y1, x2, y2 = box.chunk(4, 1)

        return torch.cat((((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h, (x2 - x1) / w, (y2 - y1) / h), 1)

    else:
        x1, y1, x2, y2 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]

        return np.stack((((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h, (x2 - x1) / w, (y2 - y1) / h), 1)


def bbox_iou(box1, box2, foreach=False, type='IoU', eps=1e-7):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, 1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, 1)

    if foreach:
        b1_x1 = b1_x1.unsqueeze(2)
        b1_y1 = b1_y1.unsqueeze(2)
        b1_x2 = b1_x2.unsqueeze(2)
        b1_y2 = b1_y2.unsqueeze(2)

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    # Intersection Area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if type == 'GIoU' or type == 'DIoU' or type == 'CIoU':
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)

        d2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
        c2 = cw ** 2 + ch ** 2 + eps

        if type == 'GIoU':
            c_area = cw * ch + eps
            iou = iou - (c_area - union) / c_area

        elif type == 'DIoU':
            iou = iou - d2 / c2

        elif type == 'CIoU':
            v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
            alpha = v / (1 - iou + v + eps)

            iou = iou - (d2 / c2 + alpha * v)

    return iou.squeeze(2) if foreach else iou


def non_max_suppression(preds, conf_t, multi_label, max_box, max_wh, iou_t, max_det, merge):
    B = preds.shape[0]
    num_cls = preds.shape[2] - 5
    candidates = preds[:, :, 4] > conf_t
    multi_label = multi_label & (num_cls > 1)

    output = [torch.zeros((0, 6), device=preds.device)] * B

    for index, pred in enumerate(preds):
        pred = pred[candidates[index]]

        if pred.shape[0] == 0:
            continue

        pred[:, 5:] *= pred[:, 4:5]

        box = xywh2xyxy(pred[:, :4])

        if multi_label:
            i, j = (pred[:, 5:] > conf_t).nonzero(as_tuple=False).T
            pred = torch.cat((box[i], pred[i, j + 5, None], j[:, None].float()), 1)

        else:
            conf, cls = pred[:, 5:].max(1, keepdim=True)
            pred = torch.cat((box, conf, cls.float()), 1)[conf.view(-1) > conf_t]

        num = pred.shape[0]
        if num == 0:
            continue

        elif num > max_box:
            pred = pred[pred[:, 4].argsort(descending=True)[:max_box]]

        scale = pred[:, 5:6] * max_wh
        boxes, confs = pred[:, :4] + scale, pred[:, 4]
        indices = torchvision.ops.nms(boxes, confs, iou_t)

        if indices.shape[0] > max_det:
            indices = indices[:max_det]

        if merge:
            iou = bbox_iou(boxes[indices], boxes, True) > iou_t
            weights = iou * confs[None]
            pred[indices, :4] = torch.mm(weights, pred[:, :4]).float() / weights.sum(1, keepdim=True)

        output[index] = pred[indices]

    return output
