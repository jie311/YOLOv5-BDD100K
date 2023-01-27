import time
import torch
import numpy as np


def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return time.time()


def to_one_hot(label, num):
    matrix = np.diag([1 for _ in range(num)])
    label = np.vectorize(lambda i: matrix[i], signature='()->(n)')(label)

    return label
