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
from itertools import product

class KBBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_candidates(self):
        pass

    @abstractmethod
    def get_all_candidates(self):
        pass

    @abstractmethod
    def logic_forward(self):
        pass
    
    @abstractmethod
    def valid_candidate(self):
        pass
    
    def _length(self, length):
        if length is None:
            length = list(self.base.keys())
        if type(length) is int:
            length = [length]
        return length
    
    def __len__(self):
        pass


class ClsKB(KBBase):
    def __init__(self, GKB_flag = False, pseudo_label_list = None, len_list = None):
        super().__init__()
        self.GKB_flag = GKB_flag
        self.pseudo_label_list = pseudo_label_list
        self.base = {}
        self.len_list = len_list
        
        if GKB_flag:
            X, Y = self.get_GKB(self.pseudo_label_list, self.len_list)
            for x, y in zip(X, Y):
                self.base.setdefault(len(x), defaultdict(list))[y].append(x)
    
    def get_GKB(self, pseudo_label_list, len_list):
        all_X = []
        for len in len_list:
            all_X += list(product(pseudo_label_list, repeat = len))
            
        X = []
        Y = []
        for x in all_X:
            if self.valid_candidate(x):
                X.append(x)
                Y.append(self.logic_forward(x))
        return X, Y
    
    def valid_candidate(self):
        pass
    
    def logic_forward(self):
        pass

    def get_candidates(self, key, length = None):
        if(self.base == {}):
            return []
        
        if key is None:
            return self.get_all_candidates()
        
        if (type(length) is int and length not in self.len_list):
            return []
        length = self._length(length)
        return sum([self.base[l][key] for l in length], [])
    
    def get_all_candidates(self):
        return sum([sum(v.values(), []) for v in self.base.values()], [])

    def _dict_len(self, dic):
        return sum(len(c) for c in dic.values())

    def __len__(self):
        return sum(self._dict_len(v) for v in self.base.values())


class add_KB(ClsKB):
    def __init__(self, GKB_flag = False, \
                    pseudo_label_list = list(range(10)), \
                    len_list = [2]):
        super().__init__(GKB_flag, pseudo_label_list, len_list)
        
    def valid_candidate(self, x):
        return True
    
    def logic_forward(self, nums):
        return sum(nums)
    
    
class hwf_KB(ClsKB):
    def __init__(self, GKB_flag = False, \
                    pseudo_label_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/'], \
                    len_list = [1, 3, 5, 7]):
        super().__init__(GKB_flag, pseudo_label_list, len_list)
        
    def valid_candidate(self, formula):
        if(len(formula) % 2 == 0):
            return False
        for i in range(len(formula)):
            if(i % 2 == 0 and formula[i] not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']):
                return False
            if(i % 2 != 0 and formula[i] not in ['+', '-', '*', '/']):
                return False
        return True
    
    def logic_forward(self, formula):
        if(self.valid_candidate(formula) == False):
            return np.inf
        try:
            return round(eval(''.join(formula)), 2)
        except ZeroDivisionError:
            return np.inf
    

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
    
              
    def get_candidates(self, key, length = None):
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
    # With ground KB
    kb = add_KB(GKB_flag = True)
    print('len(kb):', len(kb))
    res = kb.get_candidates(0)
    print(res)
    res = kb.get_candidates(18)
    print(res)
    res = kb.get_candidates(18)
    print(res)
    res = kb.get_candidates(7)
    print(res)
    print()
    
    # Without ground KB
    kb = add_KB()
    print('len(kb):', len(kb))
    res = kb.get_candidates(0)
    print(res)
    res = kb.get_candidates(18, length = 2)
    print(res)
    res = kb.get_candidates(18, length = 8)
    print(res)
    res = kb.get_candidates(7, length = 3)
    print(res)
    print()
    
    start = time.time()
    kb = hwf_KB(GKB_flag = True)
    print(time.time() - start)
    print('len(kb):', len(kb))
    res = kb.get_candidates(2, length = 1)
    print(res)
    res = kb.get_candidates(1, length = 3)
    print(res)
    res = kb.get_candidates(3.67, length = 5)
    print(res)
    print()
    
    
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

