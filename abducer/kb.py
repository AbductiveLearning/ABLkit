# coding: utf-8
# ================================================================#
#   Copyright (C) 2021 LAMDA All rights reserved.
#
#   File Name     ：kb.py
#   Author        ：freecss
#   Email         ：karlfreecss@gmail.com
#   Created Date  ：2021/06/03
#   Description   ：
#
# ================================================================#

from abc import ABC, abstractmethod
import bisect
import copy
import numpy as np

import sys

sys.path.append("..")

from collections import defaultdict
from itertools import product, combinations
from utils.utils import flatten, reform_idx, hamming_dist

from multiprocessing import Pool

import pyswip


class KBBase(ABC):
    def __init__(self, pseudo_label_list=None):
        pass

    @abstractmethod
    def logic_forward(self):
        pass

    @abstractmethod
    def abduce_candidates(self):
        pass

    def address(self, address_num, pred_res, key, multiple_predictions=False):
        new_candidates = []
        if not multiple_predictions:
            address_idx_list = list(combinations(list(range(len(pred_res))), address_num))
        else:
            address_idx_list = list(combinations(list(range(len(flatten(pred_res)))), address_num))

        for address_idx in address_idx_list:
            candidates = self.address_by_idx(pred_res, key, address_idx, multiple_predictions)
            new_candidates += candidates
        return new_candidates

    def abduction(self, pred_res, key, max_address_num, require_more_address, multiple_predictions=False):
        candidates = []

        for address_num in range(len(pred_res) + 1):
            if address_num == 0:
                if abs(self.logic_forward(pred_res) - key) <= 1e-3:
                    candidates.append(pred_res)
            else:
                new_candidates = self.address(address_num, pred_res, key, multiple_predictions)
                candidates += new_candidates

            if len(candidates) > 0:
                min_address_num = address_num
                break

            if address_num >= max_address_num:
                return [], 0, 0

        for address_num in range(min_address_num + 1, min_address_num + require_more_address + 1):
            if address_num > max_address_num:
                return candidates, min_address_num, address_num - 1
            new_candidates = self.address(address_num, pred_res, key, multiple_predictions)
            candidates += new_candidates

        return candidates, min_address_num, address_num

    def __len__(self):
        pass


class ClsKB(KBBase):
    def __init__(self, GKB_flag=False, pseudo_label_list=None, len_list=None):
        super().__init__()
        self.GKB_flag = GKB_flag
        self.pseudo_label_list = pseudo_label_list
        self.len_list = len_list
        self.prolog_flag = False

        if GKB_flag:
            self.base = {}
            X, Y = self._get_GKB(self.pseudo_label_list, self.len_list)
            for x, y in zip(X, Y):
                self.base.setdefault(len(x), defaultdict(list))[y].append(x)
        else:
            self.all_address_candidate_dict = {}
            for address_num in range(max(self.len_list) + 1):
                self.all_address_candidate_dict[address_num] = list(product(self.pseudo_label_list, repeat=address_num))

    # For parallel version of _get_GKB
    def _get_XY_list(self, args):
        pre_x, post_x_it = args[0], args[1]
        XY_list = []
        for post_x in post_x_it:
            x = (pre_x,) + post_x
            y = self.logic_forward(x)
            if y != np.inf:
                XY_list.append((x, y))
        return XY_list

    # Parallel get GKB
    def _get_GKB(self, pseudo_label_list, len_list):
        # all_X = []
        # for length in len_list:
        #     all_X += list(product(pseudo_label_list, repeat = length))

        # X, Y = [], []
        # for x in all_X:
        #     y = self.logic_forward(x)
        #     if y != np.inf:
        #         X.append(x)
        #         Y.append(y)

        X, Y = [], []
        for length in len_list:
            arg_list = []
            for pre_x in pseudo_label_list:
                post_x_it = product(pseudo_label_list, repeat=length - 1)
                arg_list.append((pre_x, post_x_it))
            with Pool(processes=len(arg_list)) as pool:
                ret_list = pool.map(self._get_XY_list, arg_list)
            for XY_list in ret_list:
                if len(XY_list) == 0:
                    continue
                part_X, part_Y = zip(*XY_list)
                X.extend(part_X)
                Y.extend(part_Y)
        return X, Y

    def logic_forward(self):
        pass

    def abduce_candidates(self, pred_res, key, max_address_num=-1, require_more_address=0, multiple_predictions=False):
        if self.GKB_flag:
            return self.abduce_from_GKB(pred_res, key, max_address_num, require_more_address)
        else:
            return self.abduction(pred_res, key, max_address_num, require_more_address, multiple_predictions)

    def abduce_from_GKB(self, pred_res, key, max_address_num, require_more_address):
        if self.base == {} or len(pred_res) not in self.len_list:
            return []

        all_candidates = self.base[len(pred_res)][key]

        if len(all_candidates) == 0:
            candidates = []
            min_address_num = 0
            address_num = 0
        else:
            cost_list = hamming_dist(pred_res, all_candidates)
            min_address_num = np.min(cost_list)
            address_num = min(max_address_num, min_address_num + require_more_address)
            idxs = np.where(cost_list <= address_num)[0]
            candidates = [all_candidates[idx] for idx in idxs]

        return candidates, min_address_num, address_num

    def address_by_idx(self, pred_res, key, address_idx, multiple_predictions=False):
        candidates = []
        abduce_c = self.all_address_candidate_dict[len(address_idx)]

        if multiple_predictions:
            save_pred_res = pred_res
            pred_res = flatten(pred_res)

        for c in abduce_c:
            candidate = pred_res.copy()
            for i, idx in enumerate(address_idx):
                candidate[idx] = c[i]

            if multiple_predictions:
                candidate = reform_idx(candidate, save_pred_res)

            if self.logic_forward(candidate) == key:
                candidates.append(candidate)
        return candidates

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
    def __init__(self, GKB_flag=False, pseudo_label_list=list(range(10)), len_list=[2]):
        super().__init__(GKB_flag, pseudo_label_list, len_list)

    def logic_forward(self, nums):
        return sum(nums)


class HWF_KB(ClsKB):
    def __init__(
        self, GKB_flag=False, pseudo_label_list=['1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', 'times', 'div'], len_list=[1, 3, 5, 7]
    ):
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
        mapping = {
            '1': '1',
            '2': '2',
            '3': '3',
            '4': '4',
            '5': '5',
            '6': '6',
            '7': '7',
            '8': '8',
            '9': '9',
            '+': '+',
            '-': '-',
            'times': '*',
            'div': '/',
        }
        formula = [mapping[f] for f in formula]
        return round(eval(''.join(formula)), 2)


class prolog_KB(KBBase):
    def __init__(self, pseudo_label_list):
        super().__init__()
        self.pseudo_label_list = pseudo_label_list
        self.prolog = pyswip.Prolog()

    def logic_forward(self):
        pass

    def abduce_candidates(self, pred_res, key, max_address_num, require_more_address, multiple_predictions):
        return self.abduction(pred_res, key, max_address_num, require_more_address, multiple_predictions)

    def address_by_idx(self, pred_res, key, address_idx, multiple_predictions=False):
        candidates = []
        # print(address_idx)
        if not multiple_predictions:
            query_string = self.get_query_string(pred_res, key, address_idx)
        else:
            query_string = self.get_query_string_need_flatten(pred_res, key, address_idx)

        if multiple_predictions:
            save_pred_res = pred_res
            pred_res = flatten(pred_res)

        abduce_c = [list(z.values()) for z in list(self.prolog.query(query_string))]
        for c in abduce_c:
            candidate = pred_res.copy()
            for i, idx in enumerate(address_idx):
                candidate[idx] = c[i]

            if multiple_predictions:
                candidate = reform_idx(candidate, save_pred_res)

            candidates.append(candidate)
        return candidates


class add_prolog_KB(prolog_KB):
    def __init__(self, pseudo_label_list=list(range(10))):
        super().__init__(pseudo_label_list)
        for i in self.pseudo_label_list:
            self.prolog.assertz("pseudo_label(%s)" % i)
        self.prolog.assertz("addition(Z1, Z2, Res) :- pseudo_label(Z1), pseudo_label(Z2), Res is Z1+Z2")

    def logic_forward(self, nums):
        return list(self.prolog.query("addition(%s, %s, Res)." % (nums[0], nums[1])))[0]['Res']

    def get_query_string(self, pred_res, key, address_idx):
        query_string = "addition("
        for idx, i in enumerate(pred_res):
            tmp = 'Z' + str(idx) + ',' if idx in address_idx else str(i) + ','
            query_string += tmp
        query_string += "%s)." % key
        return query_string


class HED_prolog_KB(prolog_KB):
    def __init__(self, pseudo_label_list=[0, 1, '+', '=']):
        super().__init__(pseudo_label_list)
        self.prolog.consult('./datasets/hed/learn_add.pl')

    # corresponding to `con_sol is not None` in `consistent_score_mapped` within `learn_add.py`
    def logic_forward(self, exs):
        return len(list(self.prolog.query("abduce_consistent_insts(%s)." % exs))) != 0

    def get_query_string_need_flatten(self, pred_res, key, address_idx):
        # flatten
        flatten_pred_res = flatten(pred_res)
        # add variables for prolog
        for idx in range(len(flatten_pred_res)):
            if idx in address_idx:
                flatten_pred_res[idx] = 'X' + str(idx)
        # unflatten
        new_pred_res = reform_idx(flatten_pred_res, pred_res)

        query_string = "abduce_consistent_insts(%s)." % new_pred_res
        return query_string.replace("'", "").replace("+", "'+'").replace("=", "'='")

    def consist_rule(self, exs, rules):
        rule_str = "%s" % rules
        rule_str = rule_str.replace("'", "")
        return len(list(self.prolog.query("consistent_inst_feature(%s, %s)." % (exs, rule_str)))) != 0

    def abduce_rules(self, pred_res):
        prolog_rules = list(self.prolog.query("consistent_inst_feature(%s, X)." % pred_res))[0]['X']
        rules = []
        for rule in prolog_rules:
            rules.append(rule.value)
        return rules

    # def consist_rules(self, pred_res, rules):


class RegKB(KBBase):
    def __init__(self, GKB_flag=False, X=None, Y=None):
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

    def abduce_candidates(self, key, length=None):
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
    t1 = time.time()
    kb = HWF_KB(True)
    t2 = time.time()
    print(t2 - t1)

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
