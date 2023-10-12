from abc import ABCMeta, abstractmethod
from typing import Any, List, Tuple

from ..learning import ABLModel
from ..reasoning import ReasonerBase


class BaseBridge(metaclass=ABCMeta):

    def __init__(self, model: ABLModel, abducer: ReasonerBase) -> None:
        if not isinstance(model, ABLModel):
            raise TypeError("Expected an ABLModel")
        if not isinstance(abducer, ReasonerBase):
            raise TypeError("Expected an ReasonerBase")
        
        self.model = model
        self.abducer = abducer

    @abstractmethod
    def predict(self, X: List[List[Any]]) -> Tuple[List[List[Any]], List[List[Any]]]:
        """Placeholder for predict labels from input."""
        pass

    @abstractmethod
    def abduce_pseudo_label(self, pseudo_label: List[List[Any]]) -> List[List[Any]]:
        """Placeholder for abduce pseudo labels."""

    @abstractmethod
    def idx_to_pseudo_label(self, idx: List[List[Any]]) -> List[List[Any]]:
        """Placeholder for map label space to symbol space."""
        pass

    @abstractmethod
    def pseudo_label_to_idx(self, pseudo_label: List[List[Any]]) -> List[List[Any]]:
        """Placeholder for map symbol space to label space."""
        pass
    
    @abstractmethod
    def train(self, train_data):
        """Placeholder for train loop of ABductive Learning."""
        pass

    @abstractmethod
    def test(self, test_data):
        """Placeholder for model test."""
        pass

    @abstractmethod
    def valid(self, valid_data):
        """Placeholder for model validation."""
        pass
    