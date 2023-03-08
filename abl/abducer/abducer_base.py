# coding: utf-8
# ================================================================#
#   Copyright (C) 2021 Freecss All rights reserved.
#
#   File Name     ：abducer_base.py
#   Author        ：freecss
#   Email         ：karlfreecss@gmail.com
#   Created Date  ：2021/06/03
#   Description   ：
#
# ================================================================#

import abc
import numpy as np
from zoopt import Dimension, Objective, Parameter, Opt
from ..utils.utils import confidence_dist, flatten, reform_idx, hamming_dist

class AbducerBase(abc.ABC):
    def __init__(self, kb, dist_func='hamming', zoopt=False, multiple_predictions=False):
        self.kb = kb
        assert dist_func == 'hamming' or dist_func == 'confidence'
        self.dist_func = dist_func
        self.zoopt = zoopt
        self.multiple_predictions = multiple_predictions
        if dist_func == 'confidence':
            self.mapping = dict(zip(self.kb.pseudo_label_list, list(range(len(self.kb.pseudo_label_list)))))

    def _get_cost_list(self, pred_res, pred_res_prob, candidates):
        if self.dist_func == 'hamming':
            if self.multiple_predictions:
                pred_res = flatten(pred_res)
                candidates = [flatten(c) for c in candidates]
                
            return hamming_dist(pred_res, candidates)
        
        elif self.dist_func == 'confidence':
            if self.multiple_predictions:
                pred_res_prob = flatten(pred_res_prob)
                candidates = [flatten(c) for c in candidates]
            candidates = [list(map(lambda x: self.mapping[x], c)) for c in candidates]
            return confidence_dist(pred_res_prob, candidates)

    def _get_one_candidate(self, pred_res, pred_res_prob, candidates):
        if len(candidates) == 0:
            return []
        elif len(candidates) == 1 or self.zoopt:
            return candidates[0]
        
        else:
            cost_list = self._get_cost_list(pred_res, pred_res_prob, candidates)
            candidate = candidates[np.argmin(cost_list)]
            return candidate

    def _zoopt_address_score(self, pred_res, pred_res_prob, key, sol): 
        if not self.multiple_predictions:
            address_idx = np.where(sol.get_x() != 0)[0]
            candidates = self.address_by_idx(pred_res, key, address_idx)
            if len(candidates) > 0:
                return np.min(self._get_cost_list(pred_res, pred_res_prob, candidates))
            else:
                return len(pred_res)
        else:
            all_address_flag = reform_idx(sol.get_x(), pred_res)
            score = 0
            for idx in range(len(pred_res)):
                address_idx = np.where(all_address_flag[idx] != 0)[0]
                candidates = self.address_by_idx([pred_res[idx]], key[idx], address_idx)
                if len(candidates) > 0:
                    score += np.min(self._get_cost_list(pred_res[idx], pred_res_prob[idx], candidates))
                else:
                    score += len(pred_res)
            return -self._zoopt_score_multiple(pred_res, key, sol.get_x())
        
    def _constrain_address_num(self, solution, max_address_num):
        x = solution.get_x()
        return max_address_num - x.sum()

    def zoopt_get_solution(self, pred_res, pred_res_prob, key, max_address_num):
        length = len(flatten(pred_res))
        dimension = Dimension(size=length, regs=[[0, 1]] * length, tys=[False] * length)
        objective = Objective(
            lambda sol: self._zoopt_address_score(pred_res, pred_res_prob, key, sol),
            dim=dimension,
            constraint=lambda sol: self._constrain_address_num(sol, max_address_num),
        )
        parameter = Parameter(budget=100, intermediate_result=False, autoset=True)
        solution = Opt.min(objective, parameter).get_x()

        return solution
    
    def address_by_idx(self, pred_res, key, address_idx):
        return self.kb.address_by_idx(pred_res, key, address_idx, self.multiple_predictions)

    def abduce(self, data, max_address_num=-1, require_more_address=0):
        pred_res, pred_res_prob, key = data
        if max_address_num == -1:
            max_address_num = len(flatten(pred_res))

        if self.zoopt:
            solution = self.zoopt_get_solution(pred_res, pred_res_prob, key, max_address_num)
            address_idx = np.where(solution != 0)[0]
            candidates = self.address_by_idx(pred_res, key, address_idx)
        else:
            candidates = self.kb.abduce_candidates(
                pred_res, key, max_address_num, require_more_address, self.multiple_predictions
            )

        candidate = self._get_one_candidate(pred_res, pred_res_prob, candidates)

        return candidate

    def abduce_rules(self, pred_res):
        return self.kb.abduce_rules(pred_res)

    def batch_abduce(self, Z, Y, max_address_num=-1, require_more_address=0):
        # if self.multiple_predictions:
        return self.abduce((Z['cls'], Z['prob'], Y), max_address_num, require_more_address)
        # else:
            # return [self.abduce((z, prob, y), max_address_num, require_more_address) for z, prob, y in zip(Z['cls'], Z['prob'], Y)]

    def __call__(self, Z, Y, max_address_num=-1, require_more_address=0):
        return self.batch_abduce(Z, Y, max_address_num, require_more_address)

if __name__ == '__main__':
    from kb import add_KB, prolog_KB, HWF_KB
    
    prob1 = [[0, 0.99, 0.01, 0, 0, 0, 0, 0, 0, 0], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
    prob2 = [[0, 0, 0.01, 0, 0, 0, 0, 0.99, 0, 0], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]

    kb = add_KB(GKB_flag=True)
    abd = AbducerBase(kb, 'confidence')
    res = abd.abduce(([1, 1], prob1, 8), max_address_num=2, require_more_address=0)
    print(res)
    res = abd.abduce(([1, 1], prob2, 8), max_address_num=2, require_more_address=0)
    print(res)
    res = abd.abduce(([1, 1], prob1, 17), max_address_num=2, require_more_address=0)
    print(res)
    res = abd.abduce(([1, 1], prob1, 17), max_address_num=1, require_more_address=0)
    print(res)
    res = abd.abduce(([1, 1], prob1, 20), max_address_num=2, require_more_address=0)
    print(res)
    print()
    
    
    multiple_prob = [[[0, 0.99, 0.01, 0, 0, 0, 0, 0, 0, 0], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]],
                     [[0, 0, 0.01, 0, 0, 0, 0, 0.99, 0, 0], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]]
    
    
    kb = add_KB()
    abd = AbducerBase(kb, 'confidence', multiple_predictions=True)
    res = abd.abduce(([[1, 1], [1, 2]], multiple_prob, [4, 8]), max_address_num=4, require_more_address=0)
    print(res)
    res = abd.abduce(([[1, 1], [1, 2]], multiple_prob, [4, 8]), max_address_num=4, require_more_address=1)
    print(res)
    print()
    

    kb = prolog_KB(pseudo_label_list=list(range(10)), pl_file='../examples/datasets/mnist_add/add.pl')
    abd = AbducerBase(kb, 'confidence')
    res = abd.abduce(([1, 1], prob1, 8), max_address_num=2, require_more_address=0)
    print(res)
    res = abd.abduce(([1, 1], prob2, 8), max_address_num=2, require_more_address=0)
    print(res)
    res = abd.abduce(([1, 1], prob1, 17), max_address_num=2, require_more_address=0)
    print(res)
    res = abd.abduce(([1, 1], prob1, 17), max_address_num=1, require_more_address=0)
    print(res)
    res = abd.abduce(([1, 1], prob1, 20), max_address_num=2, require_more_address=0)
    print(res)
    print()

    kb = prolog_KB(pseudo_label_list=list(range(10)), pl_file='../examples/datasets/mnist_add/add.pl')
    abd = AbducerBase(kb, 'confidence', zoopt=True)
    res = abd.abduce(([1, 1], prob1, 8), max_address_num=2, require_more_address=0)
    print(res)
    res = abd.abduce(([1, 1], prob2, 8), max_address_num=2, require_more_address=0)
    print(res)
    res = abd.abduce(([1, 1], prob1, 17), max_address_num=2, require_more_address=0)
    print(res)
    res = abd.abduce(([1, 1], prob1, 17), max_address_num=1, require_more_address=0)
    print(res)
    res = abd.abduce(([1, 1], prob1, 20), max_address_num=2, require_more_address=0)
    print(res)
    print()

    kb = HWF_KB(len_list=[1, 3, 5], GKB_flag=True, max_err = 0.1)
    abd = AbducerBase(kb, 'hamming')
    res = abd.abduce((['5', '+', '2'], None, 3), max_address_num=2, require_more_address=0)
    print(res)
    res = abd.abduce((['5', '+', '9'], None, 64), max_address_num=3, require_more_address=0)
    print(res)
    print()
    
    kb = HWF_KB(len_list=[1, 3, 5], GKB_flag=True, max_err = 1)
    abd = AbducerBase(kb, 'hamming')
    res = abd.abduce((['5', '+', '9'], None, 64), max_address_num=3, require_more_address=0)
    print(res)
    res = abd.abduce((['5', '+', '2'], None, 1.67), max_address_num=3, require_more_address=0)
    print(res)
    res = abd.abduce((['5', '8', '8', '8', '8'], None, 3.17), max_address_num=5, require_more_address=3)
    print(res)
    print()
    
    kb = HWF_KB(len_list=[1, 3, 5], max_err = 0.1)
    abd = AbducerBase(kb, 'hamming', multiple_predictions=True)
    res = abd.abduce(([['5', '+', '2'], ['5', '+', '9']], None, [3, 64]), max_address_num=6, require_more_address=0)
    print(res)
    print()
    
    kb = HWF_KB(len_list=[1, 3, 5], max_err = 0.1)
    abd = AbducerBase(kb, 'hamming')
    res = abd.abduce((['5', '+', '2'], None, 3), max_address_num=2, require_more_address=0)
    print(res)
    res = abd.abduce((['5', '+', '9'], None, 64), max_address_num=3, require_more_address=0)
    print(res)
    
    kb = HWF_KB(len_list=[1, 3, 5], max_err = 1)
    abd = AbducerBase(kb, 'hamming')
    res = abd.abduce((['5', '+', '9'], None, 64), max_address_num=3, require_more_address=0)
    print(res)
    res = abd.abduce((['5', '+', '2'], None, 1.67), max_address_num=3, require_more_address=0)
    print(res)
    res = abd.abduce((['5', '8', '8', '8', '8'], None, 3.17), max_address_num=5, require_more_address=3)
    print(res)
    print()

    kb = prolog_KB(pseudo_label_list=[1, 0, '+', '='], pl_file='../examples/datasets/hed/learn_add.pl')
    abd = AbducerBase(kb, zoopt=True, multiple_predictions=True)
    consist_exs = [[1, 1, '+', 0, '=', 1, 1], [1, '+', 1, '=', 1, 0], [0, '+', 0, '=', 0]]
    inconsist_exs = [[1, '+', 0, '=', 0], [1, '=', 1, '=', 0], [0, '=', 0, '=', 1, 1]]
    # inconsist_exs = [[1, '+', 0, '=', 0], ['=', '=', '=', '=', 0], ['=', '=', 0, '=', '=', '=']]
    rules = ['my_op([0], [0], [0])', 'my_op([1], [1], [1, 0])']

    print(kb._logic_forward(consist_exs, True), kb._logic_forward(inconsist_exs, True))
    print(kb.consist_rule([1, '+', 1, '=', 1, 0], rules), kb.consist_rule([1, '+', 1, '=', 1, 1], rules))
    print()

    res = abd.abduce((consist_exs, [None] * len(consist_exs), [None] * len(consist_exs)))
    print(res)
    res = abd.abduce((inconsist_exs, [None] * len(consist_exs), [None] * len(inconsist_exs)))
    print(res)
    print()

    abduced_rules = abd.abduce_rules(consist_exs)
    print(abduced_rules)