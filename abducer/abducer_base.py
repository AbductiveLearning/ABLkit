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

from itertools import product, combinations

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
    def __init__(self, kb, dist_func = "hamming", pred_res_parse = None, cache = True):
        self.kb = kb
        if dist_func == "hamming":
            dist_func = hamming_dist
        elif dist_func == "confidence":
            dist_func = confidence_dist
        self.dist_func = dist_func
        if pred_res_parse is None:
            if(dist_func == "hamming"):
                pred_res_parse = lambda x : x["cls"]
            elif dist_func == "confidence":
                pred_res_parse = lambda x : x[" "]
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
            
        if(self.kb.base != {}):
            all_candidates = self.kb.get_candidates(ans, length)
            cost_list = self.dist_func(pred_res, all_candidates)
            min_address_num = np.min(cost_list)
            address_num = min(max_address_num, min_address_num + require_more_address)
            idxs = np.where(cost_list <= address_num)[0]
            candidates = [all_candidates[idx] for idx in idxs]
            
        else:
            candidates, min_address_num, address_num = self.get_abduce_candidates(pred_res, ans, max_address_num, require_more_address)
            cost_list = self.dist_func(pred_res, candidates)
        
        if(self.cache):
            self.cache_min_address_num[(tuple(pred_res), ans)] = min_address_num
            self.cache_candidates[(tuple(pred_res), ans, address_num)] = candidates
        
        cost_list = self.dist_func(pred_res, candidates)
        min_address_num = np.min(cost_list)
        idxs = np.where(cost_list == min_address_num)[0]
        candidates = [candidates[idx] for idx in idxs]
    
        return candidates[0]
    
    def address(self, address_num, pred_res, key):
        new_candidates = []
        all_address_candidate = list(product(self.kb.pseudo_label_list, repeat = address_num))
        address_idx_list = list(combinations(list(range(len(pred_res))), address_num))
        for address_idx in address_idx_list:
            for c in all_address_candidate:
                pred_res_array = np.array(pred_res)
                if(np.count_nonzero(np.array(c) != pred_res_array[np.array(address_idx)]) == address_num):
                    pred_res_array[np.array(address_idx)] = c
                    if(abs(self.kb.logic_forward(pred_res_array) - key) <= 1e-3):
                        new_candidates.append(pred_res_array)
        return new_candidates, address_num
    
    def get_abduce_candidates(self, pred_res, key, max_address_num, require_more_address):
        
        candidates = []

        for address_num in range(len(pred_res) + 1):
            if(address_num > max_address_num):
                print('No candidates found')
                return None, None, None
            
            if(address_num == 0):
                if(abs(self.kb.logic_forward(pred_res) - key) <= 1e-3):
                    candidates.append(pred_res)
            else:
                new_candidates, address_num = self.address(address_num, pred_res, key)
                candidates += new_candidates
                
            if(len(candidates) > 0):
                min_address_num = address_num
                break
        
        for address_num in range(min_address_num + 1, min_address_num + require_more_address + 1):
            if(address_num > max_address_num):
                return candidates, min_address_num, address_num - 1
            new_candidates, address_num = self.address(address_num, pred_res, key)
            candidates += new_candidates

        return candidates, min_address_num, address_num
    
    
    
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

    
