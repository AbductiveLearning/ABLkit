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
    def __init__(self, pseudo_label_list, kb_max_len = -1):
        super().__init__()
        self.pseudo_label_list = pseudo_label_list
        self.base = {}
        self.kb_max_len = kb_max_len
        if(self.kb_max_len > 0):
            X = self.get_X(self.pseudo_label_list, self.kb_max_len)
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
        if(self.base == {}):
            return []
        
        if key is None:
            return self.get_all_candidates()

        length = self._length(length)
        if(self.kb_max_len < min(length)):
            return []
        return sum([self.base[l][key] for l in length], [])
    
    def get_all_candidates(self):
        return sum([sum(v.values(), []) for v in self.base.values()], [])
    
    def _dict_len(self, dic):
        return sum(len(c) for c in dic.values())

    def __len__(self):
        return sum(self._dict_len(v) for v in self.base.values())
    
class hwf_KB(KBBase):
    def __init__(self, pseudo_label_list, kb_max_len = -1):
        super().__init__()
        self.pseudo_label_list = pseudo_label_list
        self.base = {}
        self.kb_max_len = kb_max_len
        if(self.kb_max_len > 0):
            X = self.get_X(self.pseudo_label_list, self.kb_max_len)
            Y = self.get_Y(X, self.logic_forward)

            for x, y in zip(X, Y):
                self.base.setdefault(len(x), defaultdict(list))[y].append(np.array(x))
    
    def calculate(self, formula):
        stack = []
        postfix = []
        priority = {'+': 0, '-': 0,
                    '*': 1, '/': 1}
        skip_flag = 0
        for i in range(len(formula)):
            if formula[i] == '-':
                if i == 0:
                    formula.insert(0, 0)
        for i in range(len(formula)):
            if skip_flag:
                skip_flag -= 1
                continue
            char = formula[i]
            if char in priority.keys():
                while stack and (priority[char] <= priority[stack[-1]]):     
                    postfix.append(stack.pop())
                stack.append(char)
            else:
                num = int(char)
                while (i + 1) < len(formula):
                    if formula[i + 1] not in priority.keys():
                        skip_flag += 1
                        num = num * 10 + int(formula[i + 1])
                        i += 1
                    else:
                        break
                postfix.append(num)
        while stack:
            postfix.append(stack.pop())

        for i in postfix:
            if i in priority.keys():
                num2 = stack.pop()
                num1 = stack.pop()
                if i == '+':
                    res = num1 + num2
                elif i == '-':
                    res = num1 - num2
                elif i == '*':
                    res = num1 * num2
                elif i == '/':
                    if(num2 == 0):
                        return np.inf
                    res = num1 / num2
                stack.append(res)
            else:
                stack.append(i)
        return round(stack[0], 2)
    
    def valid_formula(self, formula):
        symbol_idx_list = []
        first_minus_flag = 0
        for idx, c in enumerate(formula):
            if(idx == 0 and c == '-'):
                first_minus_flag = 1
                continue
            if(c in ['+', '-', '*', '/']):
                if(idx - 1 in symbol_idx_list or (idx == 1 and first_minus_flag == 1)):
                    return False
                symbol_idx_list.append(idx)
        if(0 in symbol_idx_list or len(formula) - 1 in symbol_idx_list):
            return False
        return True
    
    def logic_forward(self, formula):
        if(self.valid_formula(formula) == False):
            return np.inf
        return self.calculate(list(formula))
        
    def get_X(self, pseudo_label_list, max_len):
        res = []
        assert(max_len >= 2)
        for len in range(2, max_len + 1):
            res += list(product(pseudo_label_list, repeat = len))
        return res

    def get_Y(self, X, logic_forward):
        return [logic_forward(formula) for formula in X]

    def get_candidates(self, key, length = None):
        if(self.base == {}):
            return []
        
        if key is None:
            return self.get_all_candidates()

        length = self._length(length)
        if(self.kb_max_len < min(length)):
            return []
        return sum([self.base[l][key] for l in length], [])
    
    def get_all_candidates(self):
        return sum([sum(v.values(), []) for v in self.base.values()], [])

    def _dict_len(self, dic):
        return sum(len(c) for c in dic.values())

    def __len__(self):
        return sum(self._dict_len(v) for v in self.base.values())
 
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
    # With ground KB
    pseudo_label_list = list(range(10))
    kb = add_KB(pseudo_label_list, kb_max_len = 5)
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
    
    # Without ground KB
    pseudo_label_list = list(range(10))
    kb = add_KB(pseudo_label_list)
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
    
    pseudo_label_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '+', '-', '*', '/']
    kb = hwf_KB(pseudo_label_list, kb_max_len = 5)
    print('len(kb):', len(kb))
    res = kb.get_candidates(1, length = 3)
    print(res)
    res = kb.get_candidates(3.67, length = 5)
    print(res)
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

