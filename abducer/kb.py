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

import abc
import bisect
import copy
import numpy as np

from collections import defaultdict

class KBBase(abc.ABC):
    def __init__(self, X = None, Y = None):
        pass

    def get_candidates(self, key = None, length = None):
        pass

    def get_all_candidates(self):
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
    def __init__(self, X, Y = None):
        super().__init__()
        self.base = {}
        
        if X is None:
            return

        if Y is None:
            Y = [None] * len(X)

        for x, y in zip(X, Y):
            self.base.setdefault(len(x), defaultdict(list))[y].append(np.array(x))

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

class RegKB(KBBase):
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
    X = ["1+1", "0+1", "1+0", "2+0", "1+0+1"]
    Y = [2, 1, 1, 2, 2]
    kb = ClsKB(X, Y)
    print(len(kb))
    res = kb.get_candidates(2, 5)
    print(res)
    res = kb.get_candidates(2, 3)
    print(res)
    res = kb.get_candidates(None)
    print(res)

    X = ["1+1", "0+1", "1+0", "2+0", "1+0.5", "0.75+0.75"]
    Y = [2, 1, 1, 2, 1.5, 1.5]
    kb = RegKB(X, Y)
    print(len(kb))
    res = kb.get_candidates(1.6)
    print(res)
    res = kb.get_candidates(1.6, length = 9)
    print(res)
    res = kb.get_candidates(None)
    print(res)

