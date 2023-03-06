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

from collections import defaultdict
from itertools import product, combinations
from ..utils.utils import flatten, reform_idx, hamming_dist, check_equal

from multiprocessing import Pool

import pyswip

class KBBase(ABC):
    def __init__(self, pseudo_label_list=None, len_list=None, GKB_flag=False, max_err=0):
        self.pseudo_label_list = pseudo_label_list
        self.len_list = len_list
        self.GKB_flag = GKB_flag
        self.max_err = max_err

        if GKB_flag:
            self.base = {}
            X, Y = self._get_GKB()
            for x, y in zip(X, Y):
                self.base.setdefault(len(x), defaultdict(list))[y].append(x)

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

    # Parallel _get_GKB
    def _get_GKB(self):
        X, Y = [], []
        for length in self.len_list:
            arg_list = []
            for pre_x in self.pseudo_label_list:
                post_x_it = product(self.pseudo_label_list, repeat=length - 1)
                arg_list.append((pre_x, post_x_it))
            with Pool(processes=len(arg_list)) as pool:
                ret_list = pool.map(self._get_XY_list, arg_list)
            for XY_list in ret_list:
                if len(XY_list) == 0:
                    continue
                part_X, part_Y = zip(*XY_list)
                X.extend(part_X)
                Y.extend(part_Y)
        if self.max_err != 0:      
            sorted_XY = sorted(list(zip(Y, X)))
            X = [x for y, x in sorted_XY]
            Y = [y for y, x in sorted_XY]
        return X, Y

    @abstractmethod
    def logic_forward(self):
        pass
    
    def _logic_forward(self, xs, multiple_predictions=False):
        if not multiple_predictions:
            return self.logic_forward(xs)
        else:
            res = [self.logic_forward(x) for x in xs]
            return res

    def abduce_candidates(self, pred_res, key, max_address_num=-1, require_more_address=0, multiple_predictions=False):
        if self.GKB_flag:
            return self._abduce_by_GKB(pred_res, key, max_address_num, require_more_address, multiple_predictions)
        else:
            return self._abduce_by_search(pred_res, key, max_address_num, require_more_address, multiple_predictions)
    
    @abstractmethod
    def _find_candidate_GKB(self):
        pass
    
    def _abduce_by_GKB(self, pred_res, key, max_address_num, require_more_address, multiple_predictions):
        if self.base == {}:
            return [], 0, 0

        if not multiple_predictions:
            if len(pred_res) not in self.len_list:
                return [], 0, 0
            all_candidates = self._find_candidate_GKB(pred_res, key)
            if len(all_candidates) == 0:
                return [], 0, 0
            else:
                cost_list = hamming_dist(pred_res, all_candidates)
                min_address_num = np.min(cost_list)
                address_num = min(max_address_num, min_address_num + require_more_address)
                idxs = np.where(cost_list <= address_num)[0]
                candidates = [all_candidates[idx] for idx in idxs]
                return candidates, min_address_num, address_num
       
        else:
            min_address_num = 0
            all_candidates_save = []
            cost_list_save = []
            
            for p_res, k in zip(pred_res, key):
                if len(p_res) not in self.len_list:
                    return [], 0, 0
                all_candidates = self._find_candidate_GKB(p_res, k)
                if len(all_candidates) == 0:
                    return [], 0, 0
                else:
                    all_candidates_save.append(all_candidates)
                    cost_list = hamming_dist(p_res, all_candidates)
                    min_address_num += np.min(cost_list)
                    cost_list_save.append(cost_list)
            
            multiple_all_candidates = [flatten(c) for c in product(*all_candidates_save)]
            assert len(multiple_all_candidates[0]) == len(flatten(pred_res))
            multiple_cost_list = np.array([sum(cost) for cost in product(*cost_list_save)])
            assert len(multiple_all_candidates) == len(multiple_cost_list)
            address_num = min(max_address_num, min_address_num + require_more_address)
            idxs = np.where(multiple_cost_list <= address_num)[0]
            candidates = [reform_idx(multiple_all_candidates[idx], pred_res) for idx in idxs]
            return candidates, min_address_num, address_num
    
    def address_by_idx(self, pred_res, key, address_idx, multiple_predictions=False):
        candidates = []
        abduce_c = list(product(self.pseudo_label_list, repeat=len(address_idx)))

        if multiple_predictions:
            save_pred_res = pred_res
            pred_res = flatten(pred_res)

        for c in abduce_c:
            candidate = pred_res.copy()
            for i, idx in enumerate(address_idx):
                candidate[idx] = c[i]

            if multiple_predictions:
                candidate = reform_idx(candidate, save_pred_res)

            if check_equal(self._logic_forward(candidate, multiple_predictions), key, self.max_err):
                candidates.append(candidate)
        return candidates

    def _address(self, address_num, pred_res, key, multiple_predictions):
        new_candidates = []
        if not multiple_predictions:
            address_idx_list = list(combinations(list(range(len(pred_res))), address_num))
        else:
            address_idx_list = list(combinations(list(range(len(flatten(pred_res)))), address_num))

        for address_idx in address_idx_list:
            candidates = self.address_by_idx(pred_res, key, address_idx, multiple_predictions)
            new_candidates += candidates
        return new_candidates

    def _abduce_by_search(self, pred_res, key, max_address_num, require_more_address, multiple_predictions):
        candidates = []

        for address_num in range(len(flatten(pred_res)) + 1):
            if address_num == 0:
                if check_equal(self._logic_forward(pred_res, multiple_predictions), key, self.max_err):
                    candidates.append(pred_res)
            else:
                new_candidates = self._address(address_num, pred_res, key, multiple_predictions)
                candidates += new_candidates

            if len(candidates) > 0:
                min_address_num = address_num
                break

            if address_num >= max_address_num:
                return [], 0, 0

        for address_num in range(min_address_num + 1, min_address_num + require_more_address + 1):
            if address_num > max_address_num:
                return candidates, min_address_num, address_num - 1
            new_candidates = self._address(address_num, pred_res, key, multiple_predictions)
            candidates += new_candidates

        return candidates, min_address_num, address_num

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


class ClsKB(KBBase):
    def __init__(self, pseudo_label_list, len_list, GKB_flag):
        super().__init__(pseudo_label_list, len_list, GKB_flag)

    def logic_forward(self):
        pass

    def _find_candidate_GKB(self, pred_res, key):
        return self.base[len(pred_res)][key]


class add_KB(ClsKB):
    def __init__(self, pseudo_label_list=list(range(10)), len_list=[2], GKB_flag=False):
        super().__init__(pseudo_label_list, len_list, GKB_flag)

    def logic_forward(self, nums):
        return sum(nums)


class prolog_KB(KBBase):
    def __init__(self, pseudo_label_list):
        super().__init__(pseudo_label_list)
        self.prolog = pyswip.Prolog()

    def logic_forward(self, pseudo_labels):
        result = list(self.prolog.query("logic_forward(%s, Res)." % pseudo_labels))[0]['Res']
        if result == 'true':
            return True
        elif result == 'false':
            return False
        return result
    
    def _address_pred_res(self, pred_res, address_idx, multiple_predictions):
        import re
        address_pred_res = pred_res.copy()
        if multiple_predictions:
            address_pred_res = flatten(address_pred_res)
            
        for idx in range(len(address_pred_res)):
            if idx in address_idx:
                address_pred_res[idx] = 'P' + str(idx)
        if multiple_predictions:
            address_pred_res = reform_idx(address_pred_res, pred_res)
        
        regex = r"'P\d+'"
        return re.sub(regex, lambda x: x.group().replace("'", ""), str(address_pred_res))
    
    def get_query_string(self, pred_res, key, address_idx, multiple_predictions):
        query_string = "logic_forward("
        query_string += self._address_pred_res(pred_res, address_idx, multiple_predictions)
        key_is_none_flag = key is None or (type(key) == list and key[0] is None)
        query_string += ",%s)." % key if not key_is_none_flag else ")."
        return query_string

    def _find_candidate_GKB(self):
        pass
    
    def address_by_idx(self, pred_res, key, address_idx, multiple_predictions=False):
        candidates = []
        query_string = self.get_query_string(pred_res, key, address_idx, multiple_predictions)
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


class HED_prolog_KB(prolog_KB):
    def __init__(self, pseudo_label_list=[0, 1, '+', '=']):
        super().__init__(pseudo_label_list)
        self.prolog.consult('./datasets/hed/learn_add.pl')

    def consist_rule(self, exs, rules):
        rules = str(rules).replace("\'","")
        return len(list(self.prolog.query("eval_inst_feature(%s, %s)." % (exs, rules)))) != 0

    def abduce_rules(self, pred_res):
        # print(pred_res)
        prolog_result = list(self.prolog.query("consistent_inst_feature(%s, X)." % pred_res))
        if len(prolog_result) == 0:
            return None
        prolog_rules = prolog_result[0]['X']
        rules = []
        for rule in prolog_rules:
            rules.append(rule.value)
        return rules


class RegKB(KBBase):
    def __init__(self, GKB_flag=False, pseudo_label_list=None, len_list=None, max_err=1e-3):
        super().__init__(pseudo_label_list, len_list, GKB_flag, max_err)

    def logic_forward(self):
        pass

    def _find_candidate_GKB(self, pred_res, key):
        potential_candidates = self.base[len(pred_res)]
        key_list = list(potential_candidates.keys())
        key_idx = bisect.bisect_left(key_list, key)
        
        all_candidates = []
        for idx in range(key_idx - 1, 0, -1):
            k = key_list[idx]
            if abs(k - key) <= self.max_err:
                all_candidates += potential_candidates[k]
            else:
                break
            
        for idx in range(key_idx, len(key_list)):
            k = key_list[idx]
            if abs(k - key) <= self.max_err:
                all_candidates += potential_candidates[k]
            else:
                break
        return all_candidates
    

class HWF_KB(RegKB):
    def __init__(
        self, GKB_flag=False, 
        pseudo_label_list=['1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', 'times', 'div'], 
        len_list=[1, 3, 5, 7],
        max_err=1e-3
    ):
        super().__init__(GKB_flag, pseudo_label_list, len_list, max_err)

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
        return eval(''.join(formula))


import time

if __name__ == "__main__":
    pass
    
