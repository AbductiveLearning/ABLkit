import numpy as np
from utils.plog import INFO
from collections import OrderedDict


# for multiple predictions, modify from `learn_add.py`
def flatten(l):
    return (
        [item for sublist in l for item in flatten(sublist)]
        if isinstance(l, list)
        else [l]
    )


# for multiple predictions, modify from `learn_add.py`
def reform_idx(flatten_pred_res, save_pred_res):
    re = []
    i = 0
    for e in save_pred_res:
        j = 0
        idx = []
        while j < len(e):
            idx.append(flatten_pred_res[i + j])
            j += 1
        re.append(idx)
        i = i + j
    return re


def block_sample(X, Z, Y, sample_num, epoch_idx):
    part_num = len(X) // sample_num
    if part_num == 0:
        part_num = 1
    seg_idx = epoch_idx % part_num
    INFO("seg_idx:", seg_idx, ", part num:", part_num, ", data num:", len(X))
    X = X[sample_num * seg_idx : sample_num * (seg_idx + 1)]
    Z = Z[sample_num * seg_idx : sample_num * (seg_idx + 1)]
    Y = Y[sample_num * seg_idx : sample_num * (seg_idx + 1)]

    return X, Z, Y


def hamming_dist(A, B):
    B = np.array(B)
    A = np.expand_dims(A, axis=0).repeat(axis=0, repeats=(len(B)))
    return np.sum(A != B, axis=1)


def confidence_dist(A, B):
    B = np.array(B)
    A = np.clip(A, 1e-9, 1)
    A = np.expand_dims(A, axis=0)
    A = A.repeat(axis=0, repeats=(len(B)))
    rows = np.array(range(len(B)))
    rows = np.expand_dims(rows, axis=1).repeat(axis=1, repeats=len(B[0]))
    cols = np.array(range(len(B[0])))
    cols = np.expand_dims(cols, axis=0).repeat(axis=0, repeats=len(B))
    return 1 - np.prod(A[rows, cols, B], axis=1)


def copy_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('base_model'):
            name = ".".join(k.split(".")[1:])
            new_state_dict[name] = v
    return new_state_dict
