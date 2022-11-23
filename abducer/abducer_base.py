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

import sys
sys.path.append("..")

import abc
from abducer.kb import add_KB, hwf_KB
import numpy as np

from itertools import product, combinations
import time

class AbducerBase(abc.ABC):
    def __init__(self, kb, dist_func = 'confidence', cache = True):
        self.kb = kb
        assert(dist_func == 'hamming' or dist_func == 'confidence')
        self.dist_func = dist_func        
        self.cache = cache
        
        if self.cache:
            self.cache_min_address_num = {}
            self.cache_candidates = {}

    def hamming_dist(self, A, B):
        B = np.array(B)
        A = np.expand_dims(A, axis = 0).repeat(axis=0, repeats=(len(B)))
        return np.sum(A != B, axis = 1)

    def confidence_dist(self, A, B):
        mapping = dict(zip(self.kb.pseudo_label_list, list(range(len(self.kb.pseudo_label_list)))))
        B = [list(map(lambda x : mapping[x], b)) for b in B]
    
        B = np.array(B)
        A = np.clip(A, 1e-9, 1)
        A = np.expand_dims(A, axis=0)
        A = A.repeat(axis=0, repeats=(len(B)))
        rows = np.array(range(len(B)))
        rows = np.expand_dims(rows, axis = 1).repeat(axis = 1, repeats = len(B[0]))
        cols = np.array(range(len(B[0])))
        cols = np.expand_dims(cols, axis = 0).repeat(axis = 0, repeats = len(B))
        return 1 - np.prod(A[rows, cols, B], axis = 1)
    
    
    def get_cost_list(self, pred_res, pred_res_prob, candidates):
        if self.dist_func == 'hamming':
            return self.hamming_dist(pred_res, candidates)
        elif self.dist_func == 'confidence':
            return self.confidence_dist(pred_res_prob, candidates)

    def get_min_cost_candidate(self, pred_res, pred_res_prob, candidates):
        if len(candidates) == 0:
            return []
        elif len(candidates) == 1:
            return candidates[0]
        else:
            cost_list = self.get_cost_list(pred_res, pred_res_prob, candidates)
            min_address_num = np.min(cost_list)
            idxs = np.where(cost_list == min_address_num)[0]
            return [candidates[idx] for idx in idxs][0]

    def abduce(self, data, max_address_num = -1, require_more_address = 0):
        pred_res, pred_res_prob, ans = data
        if max_address_num == -1:
            max_address_num = len(pred_res)

        if self.cache and (tuple(pred_res), ans) in self.cache_min_address_num:
            address_num = min(max_address_num, self.cache_min_address_num[(tuple(pred_res), ans)] + require_more_address)
            if (tuple(pred_res), ans, address_num) in self.cache_candidates:
                # print('cached')
                candidates = self.cache_candidates[(tuple(pred_res), ans, address_num)]
                candidate = self.get_min_cost_candidate(pred_res, pred_res_prob, candidates)
                return candidate
            
        if self.kb.GKB_flag:
            all_candidates = self.kb.get_candidates(ans, len(pred_res))
            if len(all_candidates) == 0:
                candidates = []
                min_address_num = 0
                address_num = 0
            else:
                cost_list = self.hamming_dist(pred_res, all_candidates)
                min_address_num = np.min(cost_list)
                address_num = min(max_address_num, min_address_num + require_more_address)
                idxs = np.where(cost_list <= address_num)[0]
                candidates = [all_candidates[idx] for idx in idxs]
            
        else:
            candidates, min_address_num, address_num = self.get_abduce_candidates(pred_res, ans, max_address_num, require_more_address)
        
        if self.cache:
            self.cache_min_address_num[(tuple(pred_res), ans)] = min_address_num
            self.cache_candidates[(tuple(pred_res), ans, address_num)] = candidates

        candidate = self.get_min_cost_candidate(pred_res, pred_res_prob, candidates)
        return candidate
    
    def address(self, address_num, pred_res, key):
        new_candidates = []
        all_address_candidate = list(product(self.kb.pseudo_label_list, repeat = address_num))
        address_idx_list = list(combinations(list(range(len(pred_res))), address_num))
        for address_idx in address_idx_list:
            for c in all_address_candidate:
                address_list = [pred_res[i] for i in address_idx]
                if(sum([address_list[i] == c[i] for i in range(address_num)]) == 0):
                    candidate = pred_res.copy()
                    for i, idx in enumerate(address_idx):
                        candidate[idx] = c[i]
                    if self.kb.logic_forward(candidate) == key:
                        new_candidates.append(candidate)
        return new_candidates
    
    def get_abduce_candidates(self, pred_res, key, max_address_num, require_more_address):
        candidates = []
        print(pred_res)
        for address_num in range(len(pred_res) + 1):
            if address_num == 0:
                if abs(self.kb.logic_forward(pred_res) - key) <= 1e-3:
                    candidates.append(pred_res)
            else:
                new_candidates = self.address(address_num, pred_res, key)
                candidates += new_candidates
                
            if len(candidates) > 0:
                min_address_num = address_num
                break
            
            if address_num >= max_address_num:
                return [], 0, 0
        
        for address_num in range(min_address_num + 1, min_address_num + require_more_address + 1):
            if address_num > max_address_num:
                return candidates, min_address_num, address_num - 1
            new_candidates = self.address(address_num, pred_res, key)
            candidates += new_candidates

        return candidates, min_address_num, address_num
    
     
    def batch_abduce(self, Z, Y, max_address_num = -1, require_more_address = 0):
        return [
                self.abduce((z, prob, y), max_address_num, require_more_address)\
                    for z, prob, y in zip(Z['cls'], Z['prob'], Y)
            ]

    def __call__(self, Z, Y, max_address_num = -1, require_more_address = 0):
        return self.batch_abduce(Z, Y, max_address_num, require_more_address)



if __name__ == '__main__':
    kb = add_KB(GKB_flag = True)
    abd = AbducerBase(kb, 'hamming')
    res = abd.abduce(([1, 1], None, 4), max_address_num = 2, require_more_address = 0)
    print(res)
    res = abd.abduce(([1, 1], None, 4), max_address_num = 2, require_more_address = 1)
    print(res)
    res = abd.abduce(([1, 1], None, 5), max_address_num = 2, require_more_address = 1)
    print(res)
    print()
    
    kb = hwf_KB()
    abd = AbducerBase(kb, 'hamming')
    res = abd.abduce((['5', '+', '2'], None, 3), max_address_num = 2, require_more_address = 0)
    print(res)
    res = abd.abduce((['5', '+', '2'], None, 64), max_address_num = 3, require_more_address = 0)
    print(res)
    res = abd.abduce((['5', '+', '2'], None, 1.67), max_address_num = 3, require_more_address = 0)
    print(res)
    res = abd.abduce((['5', '+', '3'], None, 0.33), max_address_num = 3, require_more_address = 3)
    print(res)
    print()
