from abc import ABC, abstractmethod
from itertools import product
from typing import Any, List, Tuple, Union

import numpy

from ..structures import ListData
from .base_kb import BaseKB


class SearchBasedKB(BaseKB, ABC):
    def __init__(
        self,
        pseudo_label_list: List,
    ) -> None:
        super().__init__(pseudo_label_list)

    @abstractmethod
    def check_equal(self, data_sample: ListData, y: Any):
        """Placeholder for check_equal."""
        pass

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

    # TODO: When the output is excessively long, use ellipses as a substitute.
    def __repr__(self):
        return (
            f"<{self.__class__.__name__}(\n"
            f"    pseudo_label_list: {self.pseudo_label_list!r}\n"
            f"    search_strategy: {self.search_strategy!r}\n"
            f"    use_cache: {self.use_cache!r}\n"
            f"    cache_root: {self.cache_root!r}\n"
            f") at {hex(id(self))}>"
        )
