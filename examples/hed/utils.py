import torch
import torch.nn as nn
import numpy as np


def gen_mappings(chars, symbs):
    n_char = len(chars)
    n_symbs = len(symbs)
    if n_char != n_symbs:
        print("Characters and symbols size dosen't match.")
        return
    from itertools import permutations

    mappings = []
    # returned mappings
    perms = permutations(symbs)
    for p in perms:
        mappings.append(dict(zip(chars, list(p))))
    return mappings


def mapping_res(original_pred_res, m):
    return [[m[symbol] for symbol in formula] for formula in original_pred_res]


def remapping_res(pred_res, m):
    remapping = {}
    for key, value in m.items():
        remapping[value] = key
    return [[remapping[symbol] for symbol in formula] for formula in pred_res]


def extract_feature(img):
    extractor = nn.AvgPool2d(2, stride=2)
    feature_map = np.array(extractor(torch.Tensor(img)))
    return feature_map.reshape((-1,))


def reduce_dimension(data):
    for truth_value in [0, 1]:
        for equation_len in range(5, 27):
            equations = data[truth_value][equation_len]
            reduced_equations = [
                [extract_feature(symbol_img) for symbol_img in equation]
                for equation in equations
            ]
            data[truth_value][equation_len] = reduced_equations 