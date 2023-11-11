from abc import ABC, abstractmethod
from itertools import combinations, product
from typing import Any, Callable, Generator, List, Optional, Tuple, Union

import numpy

from abl.structures import ListData

from ..structures import ListData
from ..utils import Cache
from .base_kb import BaseKB


def incremental_search_strategy(
    data_sample: ListData, max_revision_num: int, require_more_revision: int = 0
):
    symbol_num = data_sample["symbol_num"]
    max_revision_num = min(max_revision_num, symbol_num)
    real_end = max_revision_num
    for revision_num in range(max_revision_num + 1):
        if revision_num > real_end:
            break

        revision_idx_tuple = combinations(range(symbol_num), revision_num)
        for revision_idx in revision_idx_tuple:
            received = yield revision_idx
            if received == "success":
                real_end = min(symbol_num, revision_num + require_more_revision)


class SearchBasedKB(BaseKB, ABC):
    def __init__(
        self,
        pseudo_label_list: List,
        search_strategy: Callable[[ListData, int, int], Generator] = incremental_search_strategy,
        use_cache: bool = True,
        cache_root: Optional[str] = None,
    ) -> None:
        super().__init__(pseudo_label_list)
        self.search_strategy = search_strategy
        self.use_cache = use_cache
        if self.use_cache:
            if not hasattr(self, "get_key"):
                raise NotImplementedError("If use_cache is True, get_key should be implemented.")
            key_func = self.get_key
        else:
            key_func = lambda x: x
        self.cache = Cache[ListData, List[List[Any]]](
            func=self._abduce_by_search,
            cache=use_cache,
            cache_root=cache_root,
            key_func=key_func,
        )

    @abstractmethod
    def check_equal(self, data_sample: ListData, y: Any):
        """Placeholder for check_equal."""
        pass

    def abduce_candidates(
        self, data_sample: ListData, max_revision_num: int, require_more_revision: int = 0
    ):
        return self.cache.get(data_sample, max_revision_num, require_more_revision)

    def revise_at_idx(
        self,
        data_sample: ListData,
        revision_idx: Union[List, Tuple, numpy.ndarray],
    ):
        candidates = []
        abduce_c = product(self.pseudo_label_list, repeat=len(revision_idx))
        for c in abduce_c:
            new_data_sample = data_sample.clone()
            candidate = new_data_sample["pred_pseudo_label"][0].copy()
            for i, idx in enumerate(revision_idx):
                candidate[idx] = c[i]
            new_data_sample["pred_pseudo_label"][0] = candidate
            if self.check_equal(new_data_sample, new_data_sample["Y"][0]):
                candidates.append(candidate)
        return candidates

    def _abduce_by_search(
        self, data_sample: ListData, max_revision_num: int, require_more_revision: int = 0
    ):
        candidates = []
        gen = self.search_strategy(
            data_sample,
            max_revision_num=max_revision_num,
            require_more_revision=require_more_revision,
        )
        send_signal = True
        for revision_idx in gen:
            candidates.extend(self.revise_at_idx(data_sample, revision_idx))
            if len(candidates) > 0 and send_signal:
                try:
                    revision_idx = gen.send("success")
                    candidates.extend(self.revise_at_idx(data_sample, revision_idx))
                    send_signal = False
                except StopIteration:
                    break

        return candidates
