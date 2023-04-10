import numpy as np
from .plog import INFO
from itertools import chain

def flatten(l):
    if not isinstance(l[0], (list, tuple)):
        return l
    return list(chain.from_iterable(l))
    
def reform_idx(flatten_pred_res, save_pred_res):
    if not isinstance(save_pred_res[0], (list, tuple)):
        return flatten_pred_res
    
    re = []
    i = 0
    for e in save_pred_res:
        re.append(flatten_pred_res[i:i + len(e)])
        i += len(e)
    return re


def hamming_dist(A, B):
    A = np.array(A, dtype='<U')
    B = np.array(B, dtype='<U')
    A = np.expand_dims(A, axis = 0).repeat(axis=0, repeats=(len(B)))
    return np.sum(A != B, axis = 1)

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

def block_sample(X, Z, Y, sample_num, seg_idx):
    X = X[sample_num * seg_idx : sample_num * (seg_idx + 1)]
    Z = Z[sample_num * seg_idx : sample_num * (seg_idx + 1)]
    Y = Y[sample_num * seg_idx : sample_num * (seg_idx + 1)]
    return X, Z, Y


def check_equal(a, b, max_err=0):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= max_err
    
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        for i in range(len(a)):
            if not check_equal(a[i], b[i]):
                return False
        return True

    else:
        return a == b      
   

def to_hashable(l):
    if type(l) is not list:
        return l
    if type(l[0]) is not list:
        return tuple(l)
    return tuple(tuple(sublist) for sublist in l)

def hashable_to_list(t):
    if type(t) is not tuple:
        return t
    if type(t[0]) is not tuple:
        return list(t)
    return [list(subtuple) for subtuple in t]


def float_parameter(parameter, total_length):
    assert(type(parameter) in (int, float))
    if parameter == -1:
        return total_length
    elif type(parameter) == float:
        assert(parameter >= 0 and parameter <= 1)
        return round(total_length * parameter)
    else:
        assert(parameter >= 0)
        return parameter