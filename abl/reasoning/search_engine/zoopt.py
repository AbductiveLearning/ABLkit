from typing import List, Tuple, Union

import numpy as np
from zoopt import Dimension, Objective, Opt, Parameter, Solution

from ...structures import ListData
from ..reasoner import ReasonerBase
from ..search_based_kb import SearchBasedKB
from .base_search_engine import BaseSearchEngine


class Zoopt(BaseSearchEngine):
    def __init__(self, reasoner: ReasonerBase, kb: SearchBasedKB) -> None:
        self.reasoner = reasoner
        self.kb = kb

    def score_func(self, data_sample: ListData, solution: Solution):
        revision_idx = np.where(solution.get_x() != 0)[0]
        candidates = self.kb.revise_at_idx(data_sample, revision_idx)
        if len(candidates) > 0:
            return np.min(self.reasoner._get_dist_list(data_sample, candidates))
        else:
            return data_sample["symbol_num"]

    @staticmethod
    def constraint(solution: Solution, max_revision_num: int):
        x = solution.get_x()
        return max_revision_num - x.sum()

    def generator(
        self, data_sample: ListData, max_revision_num: int, require_more_revision: int = 0
    ) -> Union[List, Tuple, np.ndarray]:
        symbol_num = data_sample["symbol_num"]
        dimension = Dimension(size=symbol_num, regs=[[0, 1]] * symbol_num, tys=[False] * symbol_num)
        objective = Objective(
            lambda solution: self.score_func(self, data_sample, solution),
            dim=dimension,
            constraint=lambda solution: self.constraint(solution, max_revision_num),
        )
        parameter = Parameter(budget=100, intermediate_result=False, autoset=True)
        solution = Opt.min(objective, parameter).get_x()
        yield solution
