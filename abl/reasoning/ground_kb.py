from abc import ABC, abstractmethod
from typing import Any, Hashable, List

from abl.structures import ListData

from .base_kb import BaseKB


class GroundKB(BaseKB, ABC):
    def __init__(self, pseudo_label_list: List) -> None:
        super().__init__(pseudo_label_list)
        self.base = self.construct_base()

    @abstractmethod
    def construct_base(self) -> dict:
        pass

    @abstractmethod
    def get_key(self, data_sample: ListData) -> Hashable:
        pass

    def key2candidates(self, key: Hashable) -> List[List[Any]]:
        return self.base[key]

    def filter_candidates(
        self,
        data_sample: ListData,
        candidates: List[List[Any]],
        max_revision_num: int,
        require_more_revision: int = 0,
    ) -> List[List[Any]]:
        return candidates

    def abduce_candidates(
        self, data_sample: ListData, max_revision_num: int, require_more_revision: int = 0
    ):
        return self._abduce_by_GKB(
            data_sample=data_sample,
            max_revision_num=max_revision_num,
            require_more_revision=require_more_revision,
        )

    def _abduce_by_GKB(
        self, data_sample: ListData, max_revision_num: int, require_more_revision: int = 0
    ):
        candidates = self.key2candidates(self.get_key(data_sample))
        return self.filter_candidates(
            data_sample=data_sample,
            max_revision_num=max_revision_num,
            require_more_revision=require_more_revision,
            candidates=candidates,
        )
