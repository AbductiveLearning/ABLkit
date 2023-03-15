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
from ..utils.utils import confidence_dist, flatten, reform_idx, hamming_dist, nested_length

class AbducerBase(abc.ABC):
    def __init__(self, kb, dist_func='hamming', zoopt=False):
        self.kb = kb
        assert dist_func == 'hamming' or dist_func == 'confidence'
        self.dist_func = dist_func
        self.zoopt = zoopt
        if dist_func == 'confidence':
            self.mapping = dict(zip(self.kb.pseudo_label_list, list(range(len(self.kb.pseudo_label_list)))))

    def _get_cost_list(self, pred_res, pred_res_prob, candidates):
        if self.dist_func == 'hamming':
            return hamming_dist(pred_res, candidates)
        
        elif self.dist_func == 'confidence':
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
    
    def _zoopt_address_score_single(self, sol_x, pred_res, pred_res_prob, key):
        address_idx = np.where(sol_x != 0)[0]
        candidates = self.address_by_idx(pred_res, key, address_idx)
        if len(candidates) > 0:
            return np.min(self._get_cost_list(pred_res, pred_res_prob, candidates))
        else:
            return len(pred_res)
    
    def _zoopt_address_score(self, pred_res, pred_res_prob, key, sol): 
        all_address_flag = reform_idx(sol.get_x(), pred_res)
        if nested_length(pred_res) == 1:
            return self._zoopt_address_score_single(all_address_flag, pred_res, pred_res_prob, key)
        else:
            score = 0
            for idx in range(nested_length(pred_res)):
                score += self._zoopt_address_score_single(all_address_flag[idx], [pred_res[idx]], [pred_res_prob[idx]], [key[idx]])
            return score
        
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
        # print(pred_res, address_idx)
        return self.kb.address_by_idx(pred_res, key, address_idx)

    def abduce(self, data, max_address=-1, require_more_address=0):
        pred_res, pred_res_prob, key = data
        
        assert(type(max_address) in (int, float))
        if max_address == -1:
            max_address_num = len(flatten(pred_res))
        elif type(max_address) == float:
            assert(max_address >= 0 and max_address <= 1)
            max_address_num = round(len(flatten(pred_res)) * max_address)
        else:
            assert(max_address >= 0)
            max_address_num = max_address

        if self.zoopt:
            solution = self.zoopt_get_solution(pred_res, pred_res_prob, key, max_address_num)
            address_idx = np.where(solution != 0)[0]
            candidates = self.address_by_idx(pred_res, key, address_idx)
        else:
            candidates = self.kb.abduce_candidates(pred_res, key, max_address_num, require_more_address)

        candidate = self._get_one_candidate(pred_res, pred_res_prob, candidates)
        return candidate

    def abduce_rules(self, pred_res):
        return self.kb.abduce_rules(pred_res)

    def batch_abduce(self, data, max_address=-1, require_more_address=0):
        Z1, Z2, Y = data
        return [self.abduce((z, prob, y), max_address, require_more_address) for z, prob, y in zip(Z1, Z2, Y)]

    def __call__(self, Z, Y, max_address_num=-1, require_more_address=0):
        return self.batch_abduce(Z, Y, max_address_num, require_more_address)

if __name__ == '__main__':
    from kb import add_KB, prolog_KB, HWF_KB, HED_prolog_KB
    
    prob1 = [[[0, 0.99, 0.01, 0, 0, 0, 0, 0, 0, 0], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]]
    prob2 = [[[0, 0, 0.01, 0, 0, 0, 0, 0.99, 0, 0], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]]

    print('add_KB with GKB:')
    kb = add_KB(GKB_flag=True)
    abd = AbducerBase(kb, 'confidence')
    res = abd.batch_abduce(([[1, 1]], prob1, [8]), max_address=2, require_more_address=0)
    print(res)
    res = abd.batch_abduce(([[1, 1]], prob2, [8]), max_address=2, require_more_address=0)
    print(res)
    res = abd.batch_abduce(([[1, 1]], prob1, [17]), max_address=2, require_more_address=0)
    print(res)
    res = abd.batch_abduce(([[1, 1]], prob1, [17]), max_address=1, require_more_address=0)
    print(res)
    res = abd.batch_abduce(([[1, 1]], prob1, [20]), max_address=2, require_more_address=0)
    print(res)
    print()
    
    print('add_KB without GKB:')
    kb = add_KB()
    abd = AbducerBase(kb, 'confidence')
    res = abd.batch_abduce(([[1, 1]], prob1, [8]), max_address=2, require_more_address=0)
    print(res)
    res = abd.batch_abduce(([[1, 1]], prob2, [8]), max_address=2, require_more_address=0)
    print(res)
    res = abd.batch_abduce(([[1, 1]], prob1, [17]), max_address=2, require_more_address=0)
    print(res)
    res = abd.batch_abduce(([[1, 1]], prob1, [17]), max_address=1, require_more_address=0)
    print(res)
    res = abd.batch_abduce(([[1, 1]], prob1, [20]), max_address=2, require_more_address=0)
    print(res)
    print()
    
    print('prolog_KB with add.pl:')
    kb = prolog_KB(pseudo_label_list=list(range(10)), pl_file='../examples/datasets/mnist_add/add.pl')
    abd = AbducerBase(kb, 'confidence')
    res = abd.batch_abduce(([[1, 1]], prob1, [8]), max_address=2, require_more_address=0)
    print(res)
    res = abd.batch_abduce(([[1, 1]], prob2, [8]), max_address=2, require_more_address=0)
    print(res)
    res = abd.batch_abduce(([[1, 1]], prob1, [17]), max_address=2, require_more_address=0)
    print(res)
    res = abd.batch_abduce(([[1, 1]], prob1, [17]), max_address=1, require_more_address=0)
    print(res)
    res = abd.batch_abduce(([[1, 1]], prob1, [20]), max_address=2, require_more_address=0)
    print(res)
    print()

    print('prolog_KB with add.pl using zoopt:')
    kb = prolog_KB(pseudo_label_list=list(range(10)), pl_file='../examples/datasets/mnist_add/add.pl')
    abd = AbducerBase(kb, 'confidence', zoopt=True)
    res = abd.batch_abduce(([[1, 1]], prob1, [8]), max_address=2, require_more_address=0)
    print(res)
    res = abd.batch_abduce(([[1, 1]], prob2, [8]), max_address=2, require_more_address=0)
    print(res)
    res = abd.batch_abduce(([[1, 1]], prob1, [17]), max_address=2, require_more_address=0)
    print(res)
    res = abd.batch_abduce(([[1, 1]], prob1, [17]), max_address=1, require_more_address=0)
    print(res)
    res = abd.batch_abduce(([[1, 1]], prob1, [20]), max_address=2, require_more_address=0)
    print(res)
    print()
    
    print('add_KB with multiple inputs at once:')
    multiple_prob = [[[0, 0.99, 0.01, 0, 0, 0, 0, 0, 0, 0], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]],
                     [[0, 0, 0.01, 0, 0, 0, 0, 0.99, 0, 0], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]]
    
    kb = add_KB()
    abd = AbducerBase(kb, 'confidence')
    res = abd.batch_abduce(([[1, 1], [1, 2]], multiple_prob, [4, 8]), max_address=4, require_more_address=0)
    print(res)
    res = abd.batch_abduce(([[1, 1], [1, 2]], multiple_prob, [4, 8]), max_address=4, require_more_address=1)
    print(res)
    print()
    
    print('HWF_KB with GKB, max_err=0.1')
    kb = HWF_KB(len_list=[1, 3, 5], GKB_flag=True, max_err = 0.1)
    abd = AbducerBase(kb, 'hamming')
    res = abd.batch_abduce(([['5', '+', '2']], [None], [3]), max_address=2, require_more_address=0)
    print(res)
    res = abd.batch_abduce(([['5', '+', '9']], [None], [65]), max_address=3, require_more_address=0)
    print(res)
    res = abd.batch_abduce(([['5', '8', '8', '8', '8']], [None], [3.17]), max_address=5, require_more_address=3)
    print(res)
    print()
    
    print('HWF_KB without GKB, max_err=0.1')
    kb = HWF_KB(len_list=[1, 3, 5], max_err = 0.1)
    abd = AbducerBase(kb, 'hamming')
    res = abd.batch_abduce(([['5', '+', '2']], [None], [3]), max_address=2, require_more_address=0)
    print(res)
    res = abd.batch_abduce(([['5', '+', '9']], [None], [65]), max_address=3, require_more_address=0)
    print(res)
    res = abd.batch_abduce(([['5', '8', '8', '8', '8']], [None], [3.17]), max_address=5, require_more_address=3)
    print(res)
    print()
    
    print('HWF_KB with GKB, max_err=1')
    kb = HWF_KB(len_list=[1, 3, 5], GKB_flag=True, max_err = 1)
    abd = AbducerBase(kb, 'hamming')
    res = abd.batch_abduce(([['5', '+', '9']], [None], [65]), max_address=3, require_more_address=0)
    print(res)
    res = abd.batch_abduce(([['5', '+', '2']], [None], [1.67]), max_address=3, require_more_address=0)
    print(res)
    res = abd.batch_abduce(([['5', '8', '8', '8', '8']], [None], [3.17]), max_address=5, require_more_address=3)
    print(res)
    print()
    
    print('HWF_KB without GKB, max_err=1')
    kb = HWF_KB(len_list=[1, 3, 5], max_err = 1)
    abd = AbducerBase(kb, 'hamming')
    res = abd.batch_abduce(([['5', '+', '9']], [None], [65]), max_address=3, require_more_address=0)
    print(res)
    res = abd.batch_abduce(([['5', '+', '2']], [None], [1.67]), max_address=3, require_more_address=0)
    print(res)
    res = abd.batch_abduce(([['5', '8', '8', '8', '8']], [None], [3.17]), max_address=5, require_more_address=3)
    print(res)
    print()
    
    print('HWF_KB with multiple inputs at once:')
    kb = HWF_KB(len_list=[1, 3, 5], max_err = 0.1)
    abd = AbducerBase(kb, 'hamming')
    res = abd.batch_abduce(([['5', '+', '2'], ['5', '+', '9']], [None, None], [3, 64]), max_address=1, require_more_address=0)
    print(res)
    res = abd.batch_abduce(([['5', '+', '2'], ['5', '+', '9']], [None, None], [3, 64]), max_address=3, require_more_address=0)
    print(res)
    res = abd.batch_abduce(([['5', '+', '2'], ['5', '+', '9']], [None, None], [3, 65]), max_address=3, require_more_address=0)
    print(res)
    print()
    print('max_address is float')
    res = abd.batch_abduce(([['5', '+', '2'], ['5', '+', '9']], [None, None], [3, 64]), max_address=0.5, require_more_address=0)
    print(res)
    res = abd.batch_abduce(([['5', '+', '2'], ['5', '+', '9']], [None, None], [3, 64]), max_address=0.9, require_more_address=0)
    print(res)
    print()

    kb = HED_prolog_KB(pseudo_label_list=[1, 0, '+', '='], pl_file='../examples/datasets/hed/learn_add.pl')
    abd = AbducerBase(kb, zoopt=True)
    consist_exs = [[1, 1, '+', 0, '=', 1, 1], [1, '+', 1, '=', 1, 0], [0, '+', 0, '=', 0]]
    inconsist_exs = [[1, '+', 0, '=', 0], [1, '=', 1, '=', 0], [0, '=', 0, '=', 1, 1]]
    rules = ['my_op([0], [0], [0])', 'my_op([1], [1], [1, 0])']

    print(kb.logic_forward(consist_exs))
    print(kb.logic_forward(inconsist_exs))
    print()
    print(kb.consist_rule([1, '+', 1, '=', 1, 0], rules))
    print(kb.consist_rule([1, '+', 1, '=', 1, 1], rules))
    print()

    res = abd.abduce((consist_exs, [[[None]]] * len(consist_exs), [None] * len(consist_exs)))
    print(res)
    res = abd.abduce((inconsist_exs, [[[None]]] * len(consist_exs), [None] * len(inconsist_exs)))
    print(res)
    print()

    abduced_rules = abd.abduce_rules(consist_exs)
    print(abduced_rules)
