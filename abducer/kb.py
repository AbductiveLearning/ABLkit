# coding: utf-8
#================================================================#
#   Copyright (C) 2021 LAMDA All rights reserved.
#   
#   File Name     ：kb.py
#   Author        ：freecss
#   Email         ：karlfreecss@gmail.com
#   Created Date  ：2021/06/03
#   Description   ：
#
#================================================================#

from abc import ABC, abstractmethod
import bisect
import copy
import numpy as np

from collections import defaultdict
from itertools import product, combinations

import pyswip

class KBBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def logic_forward(self):
        pass
    
    @abstractmethod
    def abduce_candidates(self):
        pass
    
    
    def abduction(self, pred_res, key, max_address_num, require_more_address):
        candidates = []
        for address_num in range(len(pred_res) + 1):
            if address_num == 0:
                if abs(self.logic_forward(pred_res) - key) <= 1e-3:
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
    
    
    def __len__(self):
        pass

class ClsKB(KBBase):
    def __init__(self, GKB_flag = False, pseudo_label_list = None, len_list = None):
        super().__init__()
        self.GKB_flag = GKB_flag
        self.pseudo_label_list = pseudo_label_list
        self.len_list = len_list
        self.prolog_flag = False
        
        if GKB_flag:
            # self.base = np.load('abducer/hwf.npy', allow_pickle=True).item()
            self.base = {}
            X, Y = self.get_GKB(self.pseudo_label_list, self.len_list)
            for x, y in zip(X, Y):
                self.base.setdefault(len(x), defaultdict(list))[y].append(x)
        else:
            self.all_address_candidate_dict = {}
            for address_num in range(1, max(self.len_list) + 1):
                self.all_address_candidate_dict[address_num] = list(product(self.pseudo_label_list, repeat = address_num))
    
    def get_GKB(self, pseudo_label_list, len_list):
        all_X = []
        for len in len_list:
            all_X += list(product(pseudo_label_list, repeat = len))
            
        X = []
        Y = []
        for x in all_X:
            y = self.logic_forward(x)
            if y != np.inf:
                X.append(x)
                Y.append(y)
        return X, Y
    
    def logic_forward(self):
        pass
    
    def abduce_candidates(self, pred_res, key, max_address_num = -1, require_more_address = 0):
        if self.GKB_flag:
            return self.abduce_from_GKB(pred_res, key, max_address_num, require_more_address)
        else:
            return self.abduction(pred_res, key, max_address_num, require_more_address)



    def abduce_from_GKB(self, pred_res, key, max_address_num, require_more_address):
        if self.base == {} or len(pred_res) not in self.len_list:
            return []
        
        all_candidates = self.base[len(pred_res)][key]
        
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
            
        return candidates, min_address_num, address_num
    
    def hamming_dist(self, A, B):
        B = np.array(B)
        A = np.expand_dims(A, axis = 0).repeat(axis=0, repeats=(len(B)))
        return np.sum(A != B, axis = 1)
      

 
    def address(self, address_num, pred_res, key):  
        new_candidates = []
        all_address_candidate = self.all_address_candidate_dict[address_num]
        address_idx_list = list(combinations(list(range(len(pred_res))), address_num))
        for address_idx in address_idx_list:
            for c in all_address_candidate:
                candidate = pred_res.copy()
                for i, idx in enumerate(address_idx):
                    candidate[idx] = c[i]
                if self.logic_forward(candidate) == key:
                    if(sum(pred_res[idx] != candidate[idx] for idx in range(len(pred_res))) == address_num):
                        new_candidates.append(candidate)
        return new_candidates
    
    

    def _dict_len(self, dic):
        if not self.GKB_flag:
            return 0
        else:
            return sum(len(c) for c in dic.values())

    def __len__(self):
        if not self.GKB_flag:
            return 0
        else:
            return sum(self._dict_len(v) for v in self.base.values())


class add_KB(ClsKB):
    def __init__(self, GKB_flag = False, \
                    pseudo_label_list = list(range(10)), \
                    len_list = [2]):
        super().__init__(GKB_flag, pseudo_label_list, len_list)
    
    def logic_forward(self, nums):
        return sum(nums)
    
class hwf_KB(ClsKB):
    def __init__(self, GKB_flag = False, \
                    pseudo_label_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', 'times', 'div'], \
                    len_list = [1, 3, 5, 7]):
        super().__init__(GKB_flag, pseudo_label_list, len_list)
        
    def valid_candidate(self, formula):
        if len(formula) % 2 == 0:
            return False
        for i in range(len(formula)):
            if i % 2 == 0 and formula[i] not in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                return False
            if i % 2 != 0 and formula[i] not in ['+', '-', 'times', 'div']:
                return False
        return True
    
    def logic_forward(self, formula):
        if not self.valid_candidate(formula):
            return np.inf
        mapping = {'1':'1', '2':'2', '3':'3', '4':'4', '5':'5', '6':'6', '7':'7', '8':'8', '9':'9', '+':'+', '-':'-', 'times':'*', 'div':'/'}
        formula = [mapping[f] for f in formula]
        return round(eval(''.join(formula)), 2)


class prolog_KB(KBBase):
    def __init__(self, pseudo_label_list):
        super().__init__()
        self.pseudo_label_list = pseudo_label_list
        self.prolog = pyswip.Prolog()
        for i in self.pseudo_label_list:
            self.prolog.assertz("pseudo_label(%s)" % i)
    
    def logic_forward(self):
        pass
    
    def abduce_candidates(self, pred_res, key, max_address_num, require_more_address):
        return self.abduction(pred_res, key, max_address_num, require_more_address)
 
    def address(self, address_num, pred_res, key):      
        new_candidates = []
        address_idx_list = list(combinations(list(range(len(pred_res))), address_num))
        for address_idx in address_idx_list:
            query_string = self.get_query_string(pred_res, address_idx)
            abduce_c = [list(z.values()) for z in list(self.prolog.query(query_string % key))]
            for c in abduce_c:
                candidate = pred_res.copy()
                for i, idx in enumerate(address_idx):
                    candidate[idx] = c[i]
                if(sum(pred_res[idx] != candidate[idx] for idx in range(len(pred_res))) == address_num):
                    new_candidates.append(candidate)
        return new_candidates

    
class add_prolog_KB(prolog_KB):
    def __init__(self, pseudo_label_list = list(range(10))):
        super().__init__(pseudo_label_list)
        self.prolog.assertz("addition(Z1, Z2, Res) :- pseudo_label(Z1), pseudo_label(Z2), Res is Z1+Z2")
    
    def logic_forward(self, nums):
        return list(self.prolog.query("addition(%s, %s, Res)." %(nums[0], nums[1])))[0]['Res']
    
    def get_query_string(self, pred_res, address_idx):
        query_string = "addition("
        for idx, i in enumerate(pred_res):
            tmp = 'Z' + str(idx) + ',' if idx in address_idx else str(i) + ','
            query_string += tmp
        query_string += "%s)."
        return query_string
    


class RegKB(KBBase):
    def __init__(self, GKB_flag = False, X = None, Y = None):
        super().__init__()
        tmp_dict = {}
        for x, y in zip(X, Y):
            tmp_dict.setdefault(len(x), defaultdict(list))[y].append(np.array(x))

        self.base = {}
        for l in tmp_dict.keys():
            data = sorted(list(zip(tmp_dict[l].keys(), tmp_dict[l].values())))
            X = [x for y, x in data]
            Y = [y for y, x in data]
            self.base[l] = (X, Y)

    def valid_candidate(self):
        pass
    
    def logic_forward(self):
        pass
    
              
    def abduce_candidates(self, key, length = None):
        if key is None:
            return self.get_all_candidates()

        length = self._length(length)

        min_err = 999999
        candidates = []
        for l in length:
            X, Y = self.base[l]

            idx = bisect.bisect_left(Y, key)
            begin = max(0, idx - 1)
            end = min(idx + 2, len(X))

            for idx in range(begin, end):
                err = abs(Y[idx] - key)
                if abs(err - min_err) < 1e-9:
                    candidates.extend(X[idx])
                elif err < min_err:
                    candidates = copy.deepcopy(X[idx])
                    min_err = err
        return candidates

    def get_all_candidates(self):
        return sum([sum(D[0], []) for D in self.base.values()], [])

    def __len__(self):
        return sum([sum(len(x) for x in D[0]) for D in self.base.values()])

import time
if __name__ == "__main__":
    pass
    
    
    # X = ["1+1", "0+1", "1+0", "2+0", "1+0+1"]
    # Y = [2, 1, 1, 2, 2]
    # kb = ClsKB(X, Y)
    # print('len(kb):', len(kb))
    # res = kb.get_candidates(2, 5)
    # print(res)
    # res = kb.get_candidates(2, 3)
    # print(res)
    # res = kb.get_candidates(None)
    # print(res)
    # print()
    
    # X = ["1+1", "0+1", "1+0", "2+0", "1+0.5", "0.75+0.75"]
    # Y = [2, 1, 1, 2, 1.5, 1.5]
    # kb = RegKB(X, Y)
    # print('len(kb):', len(kb))
    # res = kb.get_candidates(1.6)
    # print(res)
    # res = kb.get_candidates(1.6, length = 9)
    # print(res)
    # res = kb.get_candidates(None)
    # print(res)

