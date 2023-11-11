from abc import ABC, abstractmethod

from ..structures import ListData


class BaseKB(ABC):
    def __init__(self, pseudo_label_list) -> None:
        self.pseudo_label_list = pseudo_label_list

    @abstractmethod
    def abduce_candidates(self, data_sample: ListData):
        """Placeholder for abduction of the knowledge base."""
        pass

    # TODO: When the output is excessively long, use ellipses as a substitute.
    def __repr__(self):
        return (
            f"<{self.__class__.__name__}(\n"
            f"    pseudo_label_list: {self.pseudo_label_list!r}\n"
            f") at {hex(id(self))}>"
        )
