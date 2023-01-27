import torch
import random
import numpy as np
from tqdm import tqdm
from scipy.cluster.vq import kmeans


def metric(anchors, whs, ratio_t):
    ratio = whs[:, None] / anchors[None]
    ratio = torch.min(ratio, 1 / ratio).min(2)[0]

    # best possible recall
    best_ratio = ratio.max(1)[0]
    bpr = (best_ratio > ratio_t).float().mean()

    # best possible score
    bps = (best_ratio * (best_ratio > ratio_t).float()).mean()

    return bpr, bps


def build_anchors(dataset, size, num, box_t, anchor_t, iter, gen, mp, sigma):
    shapes = size * dataset.shapes / dataset.shapes.max(1, keepdims=True)

    wh0s = np.concatenate([label[:, 3:5] * shape for shape, label in zip(shapes, dataset.labels)])
    wh1s = wh0s[(wh0s >= box_t).any(1)]

    std = wh1s.std(0)
    anchor0s, distortion = kmeans(wh1s / std, num, iter)
    assert len(anchor0s) == num
    anchor0s *= std

    wh0s = torch.tensor(wh0s, dtype=torch.float32)
    wh1s = torch.tensor(wh1s, dtype=torch.float32)

    bpr0, bps0 = metric(anchor0s, wh1s, 1 / anchor_t)

    print(f'build anchors: k-means anchors, bpr={bpr0}, bps={bps0}')

    # evolve
    npr = np.random
    shape = anchor0s.shape
    pbar = tqdm(range(gen), desc='evolving anchors with genetic algorithm:')
    for _ in pbar:
        mutations = np.ones(shape)
        while (mutations == 1).all():
            mutations = ((npr.random(shape) < mp) * random.random() * npr.randn(*shape) * sigma + 1).clip(0.3, 3.0)

        anchor1s = (anchor0s.copy() * mutations).clip(min=box_t)

        bpr1, bps1 = metric(anchor1s, wh1s, 1 / anchor_t)

        if bps1 > bps0:
            bpr0, bps0, anchor0s = bpr1, bps1, anchor1s

    print(f'build anchors: evolve anchors, bpr={bpr0}, bps={bps0}')

    anchors = np.round(anchor0s)
    anchors = anchors[np.argsort(anchors.prod(1))]

    return anchors


def check_anchors(dataset, model, size, box_t, bps_t, anchor_t, iter, gen, mp, sigma, save_file):
    anchors = model.head.anchors.clone().cpu().view(-1, 2)
    num = len(anchors)

    shapes = size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    scales = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))
    whs = torch.tensor(
        np.concatenate([label[:, 3:5] * shape for shape, label in zip(shapes * scales, dataset.labels)])).float()

    bpr0, bps0 = metric(anchors, whs, 1 / anchor_t)
    print(f'original anchors, bpr={bpr0}, bps={bps0}')

    if bps0 < bps_t:
        try:
            anchors = build_anchors(dataset, size, num, box_t, anchor_t, iter, gen, mp, sigma)
            bpr1, bps1 = metric(anchors, whs, 1 / anchor_t)
            print(f'new anchors, bpr={bpr1}, bps={bps1}')

            if bps1 > bps0:
                anchors = torch.tensor(anchors, device=model.head.anchors.device).type_as(model.head.anchors)
                model.head.anchors[:] = anchors.clone().view_as(model.head.anchors)
                print('new anchors saved to model')
            else:
                print('original anchors is better')

        except Exception as ex:
            print(f'build anchors error: {ex}')

    # 写入anchor
    if isinstance(anchors, torch.Tensor):
        anchors = anchors.cpu().float().numpy()
        anchors = anchors.reshape(model.head.anchors.shape)

    anchors = list(anchors)
    with open(save_file, 'w', encoding='utf-8') as f:
        f.write('anchors: \n')
        for anchor in anchors:
            anchor = [list(x) for x in list(anchor)]
            anchor = ', '.join([(str(int(x[0])) + ',' + str(int(x[1]))) for x in anchor])
            f.write('[' + anchor + ']\n')
