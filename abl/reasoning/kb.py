import bisect
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import lru_cache
from itertools import combinations, product
from multiprocessing import Pool

import numpy as np
import pyswip

from ..utils.utils import (check_equal, flatten, hamming_dist,
                           hashable_to_list, reform_idx, to_hashable)


class KBBase(ABC):
    def __init__(self, pseudo_label_list, max_err=0, use_cache=True):
        # TODO：添加一下类型检查，比如
        # if not isinstance(X, (np.ndarray, spmatrix)):
        #     raise TypeError("X should be numpy array or sparse matrix")

        self.pseudo_label_list = pseudo_label_list
        self.max_err = max_err
        self.use_cache = use_cache

    @abstractmethod
    def logic_forward(self, pseudo_labels):
        pass

    def abduce_candidates(self, pred_res, y, max_revision_num, require_more_revision=0):
        if not self.use_cache:
            return self._abduce_by_search(
                pred_res, y, max_revision_num, require_more_revision
            )
        else:
            return self._abduce_by_search_cache(
                to_hashable(pred_res),
                to_hashable(y),
                max_revision_num,
                require_more_revision,
            )

    def revise_by_idx(self, pred_res, y, revision_idx):
        candidates = []
        abduce_c = product(self.pseudo_label_list, repeat=len(revision_idx))
        for c in abduce_c:
            candidate = pred_res.copy()
            for i, idx in enumerate(revision_idx):
                candidate[idx] = c[i]
            if check_equal(self.logic_forward(candidate), y, self.max_err):
                candidates.append(candidate)
        return candidates

    def _revision(self, revision_num, pred_res, y):
        new_candidates = []
        revision_idx_list = combinations(range(len(pred_res)), revision_num)

        for revision_idx in revision_idx_list:
            candidates = self.revise_by_idx(pred_res, y, revision_idx)
            new_candidates.extend(candidates)
        return new_candidates

    def _abduce_by_search(self, pred_res, y, max_revision_num, require_more_revision):
        candidates = []
        for revision_num in range(len(pred_res) + 1):
            if revision_num == 0 and check_equal(
                self.logic_forward(pred_res), y, self.max_err
            ):
                candidates.append(pred_res)
            elif revision_num > 0:
                candidates.extend(self._revision(revision_num, pred_res, y))
            if len(candidates) > 0:
                min_revision_num = revision_num
                break
            if revision_num >= max_revision_num:
                return []

        for revision_num in range(
            min_revision_num + 1, min_revision_num + require_more_revision + 1
        ):
            if revision_num > max_revision_num:
                return candidates
            candidates.extend(self._revision(revision_num, pred_res, y))
        return candidates

    @lru_cache(maxsize=None)
    def _abduce_by_search_cache(
        self, pred_res, y, max_revision_num, require_more_revision
    ):
        pred_res = hashable_to_list(pred_res)
        y = hashable_to_list(y)
        return self._abduce_by_search(
            pred_res, y, max_revision_num, require_more_revision
        )

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


class ground_KB(KBBase):
    def __init__(self, pseudo_label_list, GKB_len_list=None, max_err=0):
        super().__init__(pseudo_label_list, max_err)

        self.GKB_len_list = GKB_len_list
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
            if y is not None:
                XY_list.append((x, y))
        return XY_list

    # Parallel _get_GKB
    def _get_GKB(self):
        X, Y = [], []
        for length in self.GKB_len_list:
            print("Generating GKB of length %d" % length)
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
        return X, Y

    def abduce_candidates(self, data_sample, max_revision_num, require_more_revision=0):
        return self._abduce_by_GKB(
            data_sample, max_revision_num, require_more_revision=require_more_revision
        )

    def _find_candidate_GKB(self, cache_key, data_sample):
        y = data_sample["Y"][0]
        if self.max_err == 0:
            return self.base[cache_key][y]
        else:
            potential_candidates = self.base[cache_key]
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

    def _abduce_by_GKB(self, data_sample, max_revision_num, require_more_revision=0):
        cache_key = len(data_sample["pred_pseudo_label"][0])
        if self.base == {} or cache_key not in self.GKB_len_list:
            return []

        all_candidates = self._find_candidate_GKB(cache_key, data_sample)
        if len(all_candidates) == 0:
            return []

        cost_array = hamming_dist(data_sample["pred_pseudo_label"][0], all_candidates)
        min_revision_num = np.min(cost_array)
        revision_num = min(max_revision_num, min_revision_num + require_more_revision)
        idxs = np.where(cost_array <= revision_num)[0]
        candidates = [all_candidates[idx] for idx in idxs]
        return candidates


class prolog_KB(KBBase):
    def __init__(self, pseudo_label_list, pl_file, max_err=0):
        super().__init__(pseudo_label_list, max_err)
        self.prolog = pyswip.Prolog()
        self.prolog.consult(pl_file)

    def logic_forward(self, pseudo_labels):
        result = list(self.prolog.query("logic_forward(%s, Res)." % pseudo_labels))[0][
            "Res"
        ]
        if result == "true":
            return True
        elif result == "false":
            return False
        return result

    def _revision_pred_res(self, pred_res, revision_idx):
        import re

        revision_pred_res = pred_res.copy()
        revision_pred_res = flatten(revision_pred_res)

        for idx in revision_idx:
            revision_pred_res[idx] = "P" + str(idx)
        revision_pred_res = reform_idx(revision_pred_res, pred_res)

        # TODO：不知道有没有更简洁的方法
        regex = r"'P\d+'"
        return re.sub(
            regex, lambda x: x.group().replace("'", ""), str(revision_pred_res)
        )

    def get_query_string(self, pred_res, y, revision_idx):
        query_string = "logic_forward("
        query_string += self._revision_pred_res(pred_res, revision_idx)
        key_is_none_flag = y is None or (type(y) == list and y[0] is None)
        query_string += ",%s)." % y if not key_is_none_flag else ")."
        return query_string

    def revise_by_idx(self, pred_res, y, revision_idx):
        candidates = []
        query_string = self.get_query_string(pred_res, y, revision_idx)
        save_pred_res = pred_res
        pred_res = flatten(pred_res)
        abduce_c = [list(z.values()) for z in self.prolog.query(query_string)]
        for c in abduce_c:
            candidate = pred_res.copy()
            for i, idx in enumerate(revision_idx):
                candidate[idx] = c[i]
            candidate = reform_idx(candidate, save_pred_res)
            candidates.append(candidate)
        return candidates
