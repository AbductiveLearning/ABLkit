import bisect
from collections import defaultdict
from itertools import product
from multiprocessing import Pool
from typing import Any, Hashable, List

import numpy as np

from abl.reasoning import GroundKB
from abl.structures import ListData
from abl.utils import hamming_dist


class HWF_KB(GroundKB):
    def __init__(
        self,
        pseudo_label_list=["1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "times", "div"],
        GKB_len_list=[1, 3, 5, 7],
        max_err=1e-10,
    ):
        self.GKB_len_list = GKB_len_list
        self.max_err = max_err
        self.label2evaluable = {str(i): str(i) for i in range(1, 10)}
        self.label2evaluable.update({"+": "+", "-": "-", "times": "*", "div": "/"})
        super().__init__(pseudo_label_list)

    def logic_forward(self, data_sample: ListData):
        if not self._valid_candidate(data_sample):
            return None
        formula = data_sample["pred_pseudo_label"][0]
        formula = [self.label2evaluable[f] for f in formula]
        data_sample["Y"] = [eval("".join(formula))]
        return data_sample["Y"][0]

    def check_equal(self, data_sample: ListData, y: Any):
        if not self._valid_candidate(data_sample):
            return False
        formula = data_sample["pred_pseudo_label"][0]
        formula = [self.label2evaluable[f] for f in formula]
        return abs(eval("".join(formula)) - y) < self.max_err

    def construct_base(self) -> dict:
        X, Y = [], []
        for length in self.GKB_len_list:
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
        if Y and isinstance(Y[0], (int, float)):
            X, Y = zip(*sorted(zip(X, Y), key=lambda pair: pair[1]))
        base = {}
        for x, y in zip(X, Y):
            base.setdefault(len(x), defaultdict(list))[y].append(x)
        return base

    @staticmethod
    def get_key(data_sample: ListData) -> Hashable:
        return (data_sample["symbol_num"], data_sample["Y"][0])

    def key2candidates(self, key: Hashable) -> List[List[Any]]:
        equation_len, y = key
        if self.max_err == 0:
            return self.base[equation_len][y]
        else:
            potential_candidates = self.base[equation_len]
            key_list = list(potential_candidates.keys())
            key_idx = bisect.bisect_left(key_list, y)

            all_candidates = []
            for idx in range(key_idx - 1, -1, -1):
                k = key_list[idx]
                if abs(k - y) <= self.max_err:
                    all_candidates.extend(potential_candidates[k])
                else:
                    break

            for idx in range(key_idx, len(key_list)):
                k = key_list[idx]
                if abs(k - y) <= self.max_err:
                    all_candidates.extend(potential_candidates[k])
                else:
                    break
            return all_candidates

    def filter_candidates(
        self,
        data_sample: ListData,
        candidates: List[List[Any]],
        max_revision_num: int,
        require_more_revision: int = 0,
    ) -> List[List[Any]]:
        cost_list = hamming_dist(data_sample["pred_pseudo_label"][0], candidates)
        min_revision_num = np.min(cost_list)
        revision_num = min(max_revision_num, min_revision_num + require_more_revision)
        idxs = np.where(cost_list <= revision_num)[0]
        filtered_candidates = [candidates[idx] for idx in idxs]
        return filtered_candidates

    # TODO: change return value to List[ListData]
    def _get_XY_list(self, args):
        pre_x, post_x_it = args[0], args[1]
        XY_list = []
        for post_x in post_x_it:
            x = (pre_x,) + post_x
            data_sample = ListData(pred_pseudo_label=[x])
            y = self.logic_forward(data_sample)
            if y is not None:
                XY_list.append((x, y))
        return XY_list

    @staticmethod
    def _valid_candidate(data_sample):
        formula = data_sample["pred_pseudo_label"][0]
        if len(formula) % 2 == 0:
            return False
        for i in range(len(formula)):
            if i % 2 == 0 and formula[i] not in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                return False
            if i % 2 != 0 and formula[i] not in ["+", "-", "times", "div"]:
                return False
        return True
