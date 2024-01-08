import math
import os

import numpy as np

from ablkit.reasoning import PrologKB, Reasoner
from ablkit.utils import reform_list

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))


class HedKB(PrologKB):
    def __init__(
        self, pseudo_label_list=[1, 0, "+", "="], pl_file=os.path.join(CURRENT_DIR, "learn_add.pl")
    ):
        pl_file = pl_file.replace("\\", "/")
        super().__init__(pseudo_label_list, pl_file)
        self.learned_rules = {}

    def consist_rule(self, exs, rules):
        rules = str(rules).replace("'", "")
        return len(list(self.prolog.query("eval_inst_feature(%s, %s)." % (exs, rules)))) != 0

    def abduce_rules(self, pred_res):
        prolog_result = list(self.prolog.query("consistent_inst_feature(%s, X)." % pred_res))
        if len(prolog_result) == 0:
            return None
        prolog_rules = prolog_result[0]["X"]
        rules = [rule.value for rule in prolog_rules]
        return rules


class HedReasoner(Reasoner):
    def revise_at_idx(self, data_example):
        revision_idx = np.where(np.array(data_example.flatten("revision_flag")) != 0)[0]
        candidate = self.kb.revise_at_idx(
            data_example.pred_pseudo_label, data_example.Y, data_example.X, revision_idx
        )
        return candidate

    def zoopt_budget(self, symbol_num):
        return 200

    def zoopt_score(self, symbol_num, data_example, sol, get_score=True):
        revision_flag = reform_list(
            list(sol.get_x().astype(np.int32)), data_example.pred_pseudo_label
        )
        data_example.revision_flag = revision_flag

        lefted_idxs = [i for i in range(len(data_example.pred_idx))]
        candidate_size = []
        max_consistent_idxs = []
        while lefted_idxs:
            idxs = []
            idxs.append(lefted_idxs.pop(0))
            max_candidate_idxs = []
            found = False
            for idx in range(-1, len(data_example.pred_idx)):
                if (idx not in idxs) and (idx >= 0):
                    idxs.append(idx)
                candidates, _ = self.revise_at_idx(data_example[idxs])
                if len(candidates) == 0:
                    if len(idxs) > 1:
                        idxs.pop()
                else:
                    if len(idxs) > len(max_candidate_idxs):
                        found = True
                        max_candidate_idxs = idxs.copy()
            removed = [i for i in lefted_idxs if i in max_candidate_idxs]
            if found:
                removed.insert(0, idxs[0])
                candidate_size.append(len(removed))
                max_consistent_idxs = max_candidate_idxs.copy()
                lefted_idxs = [i for i in lefted_idxs if i not in max_candidate_idxs]
        candidate_size.sort()
        score = 0

        for i in range(0, len(candidate_size)):
            score -= math.exp(-i) * candidate_size[i]
        if get_score:
            return score
        else:
            return max_consistent_idxs

    def abduce(self, data_example):
        symbol_num = data_example.elements_num("pred_pseudo_label")
        max_revision_num = self._get_max_revision_num(self.max_revision, symbol_num)

        solution = self._zoopt_get_solution(symbol_num, data_example, max_revision_num)
        max_candidate_idxs = self.zoopt_score(symbol_num, data_example, solution, get_score=False)

        abduced_pseudo_label = [[] for _ in range(len(data_example))]

        if len(max_candidate_idxs) > 0:
            candidates, _ = self.revise_at_idx(data_example[max_candidate_idxs])
            for i, idx in enumerate(max_candidate_idxs):
                abduced_pseudo_label[idx] = candidates[0][i]
        data_example.abduced_pseudo_label = abduced_pseudo_label
        return abduced_pseudo_label

    def abduce_rules(self, pred_res):
        return self.kb.abduce_rules(pred_res)
