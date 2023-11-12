from itertools import combinations
from typing import List, Tuple, Union

import numpy

from ...structures import ListData
from .base_search_engine import BaseSearchEngine


class BFS(BaseSearchEngine):
    def __init__(self) -> None:
        pass

    def generator(
        data_sample: ListData, max_revision_num: int, require_more_revision: int = 0
    ) -> Union[List, Tuple, numpy.ndarray]:
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
