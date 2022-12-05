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
from abducer.kb import add_KB, hwf_KB, add_prolog_KB
import numpy as np
from zoopt import Dimension, Objective, Parameter, Opt

import time

class AbducerBase(abc.ABC):
    def __init__(self, kb, dist_func = 'confidence', zoopt = False, cache = True):
        self.kb = kb
        assert(dist_func == 'hamming' or dist_func == 'confidence')
        self.dist_func = dist_func        
        self.cache = cache
        self.zoopt = zoopt
        
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



    def zoopt_address_score(self, pred_res, key, address_idx):     
        candidates = self.kb.address_by_idx(pred_res, key, address_idx)
        return 0 if len(candidates) > 0 else 1
    
    def constraint_address_num(self, solution, max_address_num):
        x = solution.get_x()
        return max_address_num - x.sum()

    def zoopt_get_address_idx(self, pred_res, key, max_address_num):
        dimension = Dimension(size=len(pred_res),
                        regs=[[0, 1]] * len(pred_res),
                        tys=[False] * len(pred_res))
        objective = Objective(lambda sol: self.zoopt_address_score(pred_res, key, [idx for idx, i in enumerate(sol.get_x()) if i != 0]), 
                              dim=dimension,
                              constraint=lambda sol: self.constraint_address_num(sol, max_address_num))
        parameter = Parameter(budget=100 * dimension.get_size(), autoset=True)
        solution = Opt.min(objective, parameter).get_x()
        
        address_idx = [idx for idx, i in enumerate(solution) if i != 0]
        address_num = solution.sum()
        
        return address_idx, address_num
    



    def abduce(self, data, max_address_num = -1, require_more_address = 0):
        pred_res, pred_res_prob, key = data
        if max_address_num == -1:
            max_address_num = len(pred_res)
            
        if self.cache and (tuple(pred_res), key) in self.cache_min_address_num:
            address_num = min(max_address_num, self.cache_min_address_num[(tuple(pred_res), key)] + require_more_address)
            if (tuple(pred_res), key, address_num) in self.cache_candidates:
                candidates = self.cache_candidates[(tuple(pred_res), key, address_num)]
                return self.get_min_cost_candidate(pred_res, pred_res_prob, candidates)    
        
        if self.zoopt:
            address_idx, address_num = self.zoopt_get_address_idx(pred_res, key, max_address_num)
            candidates = self.kb.address_by_idx(pred_res, key, address_idx)
            min_address_num = address_num
        else:
            candidates, min_address_num, address_num = self.kb.abduce_candidates(pred_res, key, max_address_num, require_more_address)
            
            
        if self.cache:
            self.cache_min_address_num[(tuple(pred_res), key)] = min_address_num
            self.cache_candidates[(tuple(pred_res), key, address_num)] = candidates

        candidate = self.get_min_cost_candidate(pred_res, pred_res_prob, candidates)    
        return candidate
    
     
    def batch_abduce(self, Z, Y, max_address_num = -1, require_more_address = 0):
        return [
                self.abduce((z, prob, y), max_address_num, require_more_address)\
                    for z, prob, y in zip(Z['cls'], Z['prob'], Y)
            ]

    def __call__(self, Z, Y, max_address_num = -1, require_more_address = 0):
        return self.batch_abduce(Z, Y, max_address_num, require_more_address)
    

if __name__ == '__main__':
    prob1 = [[0, 0.99, 0.01, 0, 0, 0, 0, 0, 0, 0], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
    prob2 = [[0, 0, 0.01, 0, 0, 0, 0, 0.99, 0, 0], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
    
    kb = add_KB(GKB_flag = True)
    abd = AbducerBase(kb, 'confidence')
    res = abd.abduce(([1, 1], prob1, 8), max_address_num = 2, require_more_address = 0)
    print(res)
    res = abd.abduce(([1, 1], prob2, 8), max_address_num = 2, require_more_address = 0)
    print(res)
    res = abd.abduce(([1, 1], prob1, 17), max_address_num = 2, require_more_address = 0)
    print(res)
    res = abd.abduce(([1, 1], prob1, 17), max_address_num = 1, require_more_address = 0)
    print(res)
    res = abd.abduce(([1, 1], prob1, 20), max_address_num = 2, require_more_address = 0)
    print(res)
    print()
    
    kb = add_prolog_KB()
    abd = AbducerBase(kb, 'confidence')
    res = abd.abduce(([1, 1], prob1, 8), max_address_num = 2, require_more_address = 0)
    print(res)
    res = abd.abduce(([1, 1], prob2, 8), max_address_num = 2, require_more_address = 0)
    print(res)
    res = abd.abduce(([1, 1], prob1, 17), max_address_num = 2, require_more_address = 0)
    print(res)
    res = abd.abduce(([1, 1], prob1, 17), max_address_num = 1, require_more_address = 0)
    print(res)
    res = abd.abduce(([1, 1], prob1, 20), max_address_num = 2, require_more_address = 0)
    print(res)
    print()
    
    kb = add_prolog_KB()
    abd = AbducerBase(kb, 'confidence', zoopt = True)
    res = abd.abduce(([1, 1], prob1, 8), max_address_num = 2, require_more_address = 0)
    print(res)
    res = abd.abduce(([1, 1], prob2, 8), max_address_num = 2, require_more_address = 0)
    print(res)
    res = abd.abduce(([1, 1], prob1, 17), max_address_num = 2, require_more_address = 0)
    print(res)
    res = abd.abduce(([1, 1], prob1, 17), max_address_num = 1, require_more_address = 0)
    print(res)
    res = abd.abduce(([1, 1], prob1, 20), max_address_num = 2, require_more_address = 0)
    print(res)
    print()
    
    kb = hwf_KB(len_list = [1, 3, 5])
    abd = AbducerBase(kb, 'hamming')
    res = abd.abduce((['5', '+', '2'], None, 3), max_address_num = 2, require_more_address = 0)
    print(res)
    res = abd.abduce((['5', '+', '2'], None, 64), max_address_num = 3, require_more_address = 0)
    print(res)
    res = abd.abduce((['5', '+', '2'], None, 1.67), max_address_num = 3, require_more_address = 0)
    print(res)
    res = abd.abduce((['5', '8', '8', '8', '8'], None, 3.17), max_address_num = 5, require_more_address = 3)
    print(res)
    print()
    

