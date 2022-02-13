# coding: utf-8
#================================================================#
#   Copyright (C) 2021 Freecss All rights reserved.
#   
#   File Name     ：abducer_base.py
#   Author        ：freecss
#   Email         ：karlfreecss@gmail.com
#   Created Date  ：2021/06/03
#   Description   ：
#
#================================================================#

import abc
from abducer.kb import ClsKB, RegKB
#from kb import ClsKB, RegKB
import numpy as np

def hamming_dist(A, B):
    B = np.array(B)
    A = np.expand_dims(A, axis = 0).repeat(axis=0, repeats=(len(B)))
    return np.sum(A != B, axis = 1)

def confidence_dist(A, B):
    B = np.array(B)

    #print(A)
    A = np.clip(A, 1e-9, 1)
    A = np.expand_dims(A, axis=0)
    A = A.repeat(axis=0, repeats=(len(B)))
    rows = np.array(range(len(B)))
    rows = np.expand_dims(rows, axis = 1).repeat(axis = 1, repeats = len(B[0]))
    cols = np.array(range(len(B[0])))
    cols = np.expand_dims(cols, axis = 0).repeat(axis = 0, repeats = len(B))
    return 1 - np.prod(A[rows, cols, B], axis = 1)

class AbducerBase(abc.ABC):
    def __init__(self, kb, dist_func = "hamming", pred_res_parse = None):
        self.kb = kb
        if dist_func == "hamming":
            dist_func = hamming_dist
        elif dist_func == "confidence":
            dist_func = confidence_dist
        self.dist_func = dist_func
        if pred_res_parse is None:
            pred_res_parse = lambda x : x["cls"]
        self.pred_res_parse = pred_res_parse

    def abduce(self, data, max_address_num, require_more_address, length = -1):
        pred_res, ans = data

        if length == -1:
            length = len(pred_res)

        candidates = self.kb.get_candidates(ans, length)
        pred_res = np.array(pred_res)

        cost_list = self.dist_func(pred_res, candidates)
        address_num = np.min(cost_list)
        threshold = min(address_num + require_more_address, max_address_num)
        idxs = np.where(cost_list <= address_num+require_more_address)[0]

        #return [candidates[idx] for idx in idxs], address_num
        if len(idxs) > 1:
            return None
        return [candidates[idx] for idx in idxs][0]

    def batch_abduce(self, Y, C, max_address_num = 3, require_more_address = 0):
        return [
                self.abduce((y, c), max_address_num, require_more_address)\
                    for y, c in zip(self.pred_res_parse(Y), C)
            ]

    def __call__(self, Y, C, max_address_num = 3, require_more_address = 0):
        return batch_abduce(Y, C, max_address_num, require_more_address)

if __name__ == "__main__":
    #["1+1", "0+1", "1+0", "2+0"]
    X = [[1,3,1], [0,3,1], [1,2,0], [3,2,0]]
    Y = [2, 1, 1, 2]
    kb = RegKB(X, Y)
    
    abd = AbducerBase(kb)
    res = abd.abduce(([0,2,0], None), 1, 0)
    print(res)
    res = abd.abduce(([0, 2, 0], 0.99), 1, 0)
    print(res)

    A = np.array([[0.5, 0.25, 0.25, 0], [0.3, 0.3, 0.3, 0.1], [0.1, 0.2, 0.3, 0.4]])
    B = [[1, 2, 3], [0, 1, 3]]
    res = confidence_dist(A, B)
    print(res)

    A = np.array([[0.5, 0.25, 0.25, 0], [0.3, 1.0, 0.3, 0.1], [0.1, 0.2, 0.3, 1.0]])
    B = [[0, 1, 3]]
    res = confidence_dist(A, B)
    print(res)

    kb_str = ['10010001011', '00010001100', '00111101011', '11101000011', '11110011001', '11111010001', '10001010010', '11100100001', '10001001100', '11011010001', '00110000100', '11000000111', '01110111111', '11000101100', '10101011010', '00000110110', '11111110010', '11100101100', '10111001111', '10000101100', '01001011101', '01001110000', '01110001110', '01010010001', '10000100010', '01001011011', '11111111100', '01011101101', '00101110101', '11101001101', '10010110000', '10000000011']
    X = [[int(c) for c in s] for s in kb_str]
    kb = RegKB(X, len(X) * [None])

    abd = AbducerBase(kb)
    res = abd.abduce(((1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1), None), 1, 0)
    print(res)
