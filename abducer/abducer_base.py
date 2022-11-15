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
from kb import add_KB
import numpy as np

def hamming_dist(A, B):
    return np.sum(np.array(A) != np.array(B))

def hamming_dist_kb(A, B):
    B = np.array(B)
    A = np.expand_dims(A, axis = 0).repeat(axis=0, repeats=(len(B)))
    return np.sum(A != B, axis = 1)

def confidence_dist_kb(A, B):
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
    def __init__(self, kb, dist_func = "hamming", pred_res_parse = None, cache = True):
        self.kb = kb
        if dist_func == "hamming":
            self.dist_func = hamming_dist
        if pred_res_parse is None:
            pred_res_parse = lambda x : x["cls"]
        self.pred_res_parse = pred_res_parse
        
        self.cache = cache
        self.cache_min_address_num = {}
        self.cache_candidates = {}

    def abduce(self, data, max_address_num = 3, require_more_address = 0, length = -1):
        pred_res, ans = data

        if length == -1:
            length = len(pred_res)

        if(self.cache and (tuple(pred_res), ans) in self.cache_min_address_num):
            address_num = min(max_address_num, self.cache_min_address_num[(tuple(pred_res), ans)] + require_more_address)
            if((tuple(pred_res), ans, address_num) in self.cache_candidates):
                print('cached')
                return self.cache_candidates[(tuple(pred_res), ans, address_num)]
            
        
        candidates, min_address_num, address_num = self.kb.get_abduce_candidates(pred_res, ans, length, self.dist_func, max_address_num, require_more_address)
        
        if(self.cache):
            self.cache_min_address_num[(tuple(pred_res), ans)] = min_address_num
            self.cache_candidates[(tuple(pred_res), ans, address_num)] = candidates

        return candidates
        
        # candidates = self.kb.get_candidates(ans, length)
        
        # cost_list = self.dist_func(pred_res, candidates)
        # address_num = np.min(cost_list)
        # # threshold = min(address_num + require_more_address, max_address_num)
        # idxs = np.where(cost_list <= address_num + require_more_address)[0]

        # return [candidates[idx] for idx in idxs], address_num
    
    
    
        # if len(idxs) > 1:
        #     return None
        # return [candidates[idx] for idx in idxs]

    def batch_abduce(self, Y, C, max_address_num = 3, require_more_address = 0):
        return [
                self.abduce((y, c), max_address_num, require_more_address)\
                    for y, c in zip(self.pred_res_parse(Y), C)
            ]

    def __call__(self, Y, C, max_address_num = 3, require_more_address = 0):
        return self.batch_abduce(Y, C, max_address_num, require_more_address)



if __name__ == "__main__":
    pseudo_label_list = list(range(10))
    kb = add_KB(pseudo_label_list)
    abd = AbducerBase(kb)
    res = abd.abduce(([1, 1, 1], 4), max_address_num = 2, require_more_address = 0)
    print(res)
    print()
    res = abd.abduce(([1, 1, 1], 4), max_address_num = 2, require_more_address = 1)
    print(res)
    print()
    res = abd.abduce(([1, 1, 1], 4), max_address_num = 1, require_more_address = 1)
    print(res)
    print()
    res = abd.abduce(([1, 1, 1], 4), max_address_num = 2, require_more_address = 0)
    print(res)
    print()
    res = abd.abduce(([1, 1, 1], 5), max_address_num = 2, require_more_address = 1)
    print(res)
    # res = abd.abduce(([0, 2, 0], 0.99), 1, 0)
    # print(res)

    
