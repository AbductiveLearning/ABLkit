import numpy as np

# for multiple predictions, modify from `learn_add.py`
def flatten(l):
    return [item for sublist in l for item in _flatten(sublist)] if isinstance(l, list) else [l]
    
# for multiple predictions, modify from `learn_add.py`
def reform_ids(flatten_pred_res, save_pred_res):
    re = []
    i = 0
    for e in save_pred_res:
        j = 0
        ids = []
        while j < len(e):
            ids.append(flatten_pred_res[i + j])
            j += 1
        re.append(ids)
        i = i + j
    return re

def hamming_dist(A, B):
    B = np.array(B)
    A = np.expand_dims(A, axis = 0).repeat(axis=0, repeats=(len(B)))
    return np.sum(A != B, axis = 1)

def confidence_dist(A, B):
    B = np.array(B)
    A = np.clip(A, 1e-9, 1)
    A = np.expand_dims(A, axis=0)
    A = A.repeat(axis=0, repeats=(len(B)))
    rows = np.array(range(len(B)))
    rows = np.expand_dims(rows, axis = 1).repeat(axis = 1, repeats = len(B[0]))
    cols = np.array(range(len(B[0])))
    cols = np.expand_dims(cols, axis = 0).repeat(axis = 0, repeats = len(B))
    return 1 - np.prod(A[rows, cols, B], axis = 1)
