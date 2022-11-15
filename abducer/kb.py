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
    def logic_forward(self, X):
        pass
    
    def _length(self, length):
        if length is None:
            length = list(self.base.keys())
        if type(length) is int:
            length = [length]
        return length
    
    def __len__(self):
        pass

class add_KB(KBBase):
    def __init__(self, pseudo_label_list, max_len = 5):
        super().__init__()
        self.pseudo_label_list = pseudo_label_list
        self.base = {}
        
        X = self.get_X(self.pseudo_label_list, max_len)
        Y = self.get_Y(X, self.logic_forward)

        for x, y in zip(X, Y):
            self.base.setdefault(len(x), defaultdict(list))[y].append(np.array(x))
    
    def logic_forward(self, nums):
        return sum(nums)
    
    def get_X(self, pseudo_label_list, max_len):
        res = []
        assert(max_len >= 2)
        for len in range(2, max_len + 1):
            res += list(product(pseudo_label_list, repeat = len))
        return res

    def get_Y(self, X, logic_forward):
        return [logic_forward(nums) for nums in X]

    def get_candidates(self, key, length = None):
        if key is None:
            return self.get_all_candidates()

        length = self._length(length)
        return sum([self.base[l][key] for l in length], [])
    
    def get_all_candidates(self):
        return sum([sum(v.values(), []) for v in self.base.values()], [])
    
    def get_abduce_candidates(self, pred_res, key, length, dist_func, max_address_num, require_more_address):
        if key is None:
            return self.get_all_candidates()
        
        candidates = []
        all_candidates = list(product(self.pseudo_label_list, repeat = len(pred_res)))
        for address_num in range(length + 1):
            if(address_num > max_address_num):
                print('No candidates found')
                return None, None, None
            for c in all_candidates:
                if(dist_func(c, pred_res) == address_num):
                    if(self.logic_forward(c) == key):
                        candidates.append(c)
            if(len(candidates) > 0):
                min_address_num = address_num
                break
        
        for address_num in range(min_address_num + 1, min_address_num + require_more_address + 1):
            if(address_num > max_address_num):
                return candidates, min_address_num, address_num - 1
            for c in all_candidates:
                if(dist_func(c, pred_res) == address_num):
                    if(self.logic_forward(c) == key):
                        candidates.append(c)

        return candidates, min_address_num, address_num
        

    def _dict_len(self, dic):
        return sum(len(c) for c in dic.values())

    def __len__(self):
        return sum(self._dict_len(v) for v in self.base.values())



if __name__ == "__main__":
    pseudo_label_list = list(range(10))
    kb = add_KB(pseudo_label_list, max_len = 5)
    print('len(kb):', len(kb))
    print()
    res = kb.get_candidates(0)
    print(res)
    print()
    res = kb.get_candidates(18, length = 2)
    print(res)
    print()
    res = kb.get_candidates(7, length = 3)
    print(res)

