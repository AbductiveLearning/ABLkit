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
    
    def get_abduce_candidates(self, pred_res, key, max_address_num, require_more_address):
        if key is None:
            return self.get_all_candidates()
        
        candidates = []

        for address_num in range(len(pred_res) + 1):
            if(address_num > max_address_num):
                print('No candidates found')
                return None, None, None
            
            if(address_num == 0):
                if(self.logic_forward(pred_res) == key):
                    candidates.append(pred_res)
            else:
                all_address_candidate = list(product(self.pseudo_label_list, repeat = address_num))
                address_idx_list = list(combinations(list(range(len(pred_res))), address_num))
                for address_idx in address_idx_list:
                    for c in all_address_candidate:
                        if(np.count_nonzero(np.array(c) != np.array(pred_res)[np.array(address_idx)]) == address_num):
                            pred_res_array = np.array(pred_res)
                            pred_res_array[np.array(address_idx)] = c
                            if(self.logic_forward(pred_res_array) == key):
                                candidates.append(pred_res_array)
            
            if(len(candidates) > 0):
                min_address_num = address_num
                break
        
        for address_num in range(min_address_num + 1, min_address_num + require_more_address + 1):
            if(address_num > max_address_num):
                return candidates, min_address_num, address_num - 1
            all_candidate = list(product(self.pseudo_label_list, repeat = address_num))
            address_idx_list = list(combinations(list(range(len(pred_res))), address_num))
            for address_idx in address_idx_list:
                for c in all_candidate:
                    if(np.count_nonzero(np.array(c) != np.array(pred_res)[np.array(address_idx)]) == address_num):
                        pred_res_array = np.array(pred_res)
                        pred_res_array[np.array(address_idx)] = c
                        if(self.logic_forward(pred_res_array) == key):
                            candidates.append(pred_res_array)

        return candidates, min_address_num, address_num
        

    def _dict_len(self, dic):
        return sum(len(c) for c in dic.values())

    def __len__(self):
        return sum(self._dict_len(v) for v in self.base.values())
    
# class hwf_KB(KBBase):
#     def __init__(self, pseudo_label_list, max_len = 5):
#         super().__init__()
#         self.pseudo_label_list = pseudo_label_list
#         self.base = {}
        
#         X = self.get_X(self.pseudo_label_list, max_len)
#         Y = self.get_Y(X, self.logic_forward)

#         for x, y in zip(X, Y):
#             self.base.setdefault(len(x), defaultdict(list))[y].append(np.array(x))
    
#     def logic_forward(self, nums):
#         return sum(nums)
    
#     def get_X(self, pseudo_label_list, max_len):
#         res = []
#         assert(max_len >= 2)
#         for len in range(2, max_len + 1):
#             res += list(product(pseudo_label_list, repeat = len))
#         return res

#     def get_Y(self, X, logic_forward):
#         return [logic_forward(nums) for nums in X]

#     def get_candidates(self, key, length = None):
#         if key is None:
#             return self.get_all_candidates()

#         length = self._length(length)
#         return sum([self.base[l][key] for l in length], [])
    
#     def get_all_candidates(self):
#         return sum([sum(v.values(), []) for v in self.base.values()], [])
    
#     def get_abduce_candidates(self, pred_res, key, length, dist_func, max_address_num, require_more_address):
#         if key is None:
#             return self.get_all_candidates()
        
#         candidates = []
#         # all_candidates = list(product(self.pseudo_label_list, repeat = len(pred_res)))

#         for address_num in range(length + 1):
#             if(address_num > max_address_num):
#                 print('No candidates found')
#                 return None, None, None
            
#             if(address_num == 0):
#                 if(self.logic_forward(pred_res) == key):
#                     candidates.append(pred_res)
#             else:
#                 all_address_candidate = list(product(self.pseudo_label_list, repeat = address_num))
#                 address_idx_list = list(combinations(list(range(len(pred_res))), address_num))
#                 for address_idx in address_idx_list:
#                     for c in all_address_candidate:
#                         pred_res_array = np.array(pred_res)
#                         pred_res_array[np.array(address_idx)] = c
#                         if(np.count_nonzero(np.array(c) != np.array(pred_res)[np.array(address_idx)]) == address_num and self.logic_forward(pred_res_array) == key):
#                             candidates.append(pred_res_array)
            
#             if(len(candidates) > 0):
#                 min_address_num = address_num
#                 break
        
#         for address_num in range(min_address_num + 1, min_address_num + require_more_address + 1):
#             if(address_num > max_address_num):
#                 return candidates, min_address_num, address_num - 1
#             all_candidate = list(product(self.pseudo_label_list, repeat = address_num))
#             address_idx_list = list(combinations(list(range(len(pred_res))), address_num))
#             for address_idx in address_idx_list:
#                 for c in all_candidate:
#                     pred_res_array = np.array(pred_res)
#                     pred_res_array[np.array(address_idx)] = c
#                     if(np.count_nonzero(np.array(c) != pred_res_array[np.array(address_idx)]) == address_num and self.logic_forward(pred_res_array) == key):
#                         candidates.append(pred_res_array)

#         return candidates, min_address_num, address_num
        

#     def _dict_len(self, dic):
#         return sum(len(c) for c in dic.values())

#     def __len__(self):
#         return sum(self._dict_len(v) for v in self.base.values())

class cls_KB(KBBase):
    def __init__(self, X, Y = None):
        super().__init__()
        self.base = {}
        
        if X is None:
            return

        if Y is None:
            Y = [None] * len(X)

        for x, y in zip(X, Y):
            self.base.setdefault(len(x), defaultdict(list))[y].append(np.array(x))
    
    def logic_forward(self):
        return None

    def get_candidates(self, key, length = None):
        if key is None:
            return self.get_all_candidates()

        length = self._length(length)

        return sum([self.base[l][key] for l in length], [])
    
    def get_all_candidates(self):
        return sum([sum(v.values(), []) for v in self.base.values()], [])

    def _dict_len(self, dic):
        return sum(len(c) for c in dic.values())

    def __len__(self):
        return sum(self._dict_len(v) for v in self.base.values())

class reg_KB(KBBase):
    def __init__(self, X, Y = None):
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

    def logic_forward(self):
        return None
              
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
    print()
    
    pseudo_label_list = list(range(10)) + ['+', '-', '*', '/']
    kb = hwf_KB(pseudo_label_list, max_len = 5)
    print('len(kb):', len(kb))
    print()
    
    
    X = ["1+1", "0+1", "1+0", "2+0", "1+0+1"]
    Y = [2, 1, 1, 2, 2]
    kb = cls_KB(X, Y)
    print('len(kb):', len(kb))
    res = kb.get_candidates(2, 5)
    print(res)
    res = kb.get_candidates(2, 3)
    print(res)
    res = kb.get_candidates(None)
    print(res)
    print()
    
    X = ["1+1", "0+1", "1+0", "2+0", "1+0.5", "0.75+0.75"]
    Y = [2, 1, 1, 2, 1.5, 1.5]
    kb = reg_KB(X, Y)
    print('len(kb):', len(kb))
    res = kb.get_candidates(1.6)
    print(res)
    res = kb.get_candidates(1.6, length = 9)
    print(res)
    res = kb.get_candidates(None)
    print(res)

