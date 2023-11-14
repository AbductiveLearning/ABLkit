from abc import ABC, abstractmethod
from typing import Any, Generator, List, Tuple, Union

import numpy as np
import pyswip

from ..structures import ListData
from .base_kb import BaseKB


class PrologBasedKB(BaseKB, ABC):
    def __init__(self, pseudo_label_list, pl_file):
        self.pseudo_label_list = pseudo_label_list
        self.prolog = pyswip.Prolog()
        self.prolog.consult(pl_file)

    def logic_forward(
        self, data_sample: ListData, revision_idx: Union[List, Tuple, np.ndarray] = None
    ) -> Generator[Union[Any, pyswip.Variable, list, dict, None], Any, None]:
        return self.prolog.query(self.to_query(data_sample, revision_idx))

    @abstractmethod
    def to_query(self, data_sample: ListData, revision_idx: Union[List, Tuple, np.ndarray] = None):
        pass

    @abstractmethod
    def postprocess(
        self, query_res, data_sample: ListData, revision_idx: Union[List, Tuple, np.ndarray]
    ):
        return list(query_res)

    @abstractmethod
    def filter_candidates(
        self,
        data_sample: ListData,
        candidates: List[List[Any]],
        max_revision_num: int,
        require_more_revision: int = 0,
    ) -> List[List[Any]]:
        return candidates

    def revise_at_idx(self, data_sample: ListData, revision_idx: Union[List, Tuple, np.ndarray]):
        query_res = self.logic_forward(data_sample, revision_idx)
        return self.postprocess(query_res, data_sample, revision_idx)
