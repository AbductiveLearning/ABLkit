from abc import ABC, abstractmethod

from ..structures import ListData


class BaseKB(ABC):
    def __init__(self, pseudo_label_list) -> None:
        self.pseudo_label_list = pseudo_label_list

    @abstractmethod
    def abduce_candidates(self, data_sample: ListData):
        """Placeholder for abduction of the knowledge base."""
        pass
