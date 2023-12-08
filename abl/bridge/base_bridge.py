from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional, Tuple, Union

from ..learning import ABLModel
from ..reasoning import Reasoner
from ..structures import ListData

DataSet = Tuple[List[List[Any]], Optional[List[List[Any]]], List[List[Any]]]


class BaseBridge(metaclass=ABCMeta):
    def __init__(self, model: ABLModel, reasoner: Reasoner) -> None:
        if not isinstance(model, ABLModel):
            raise TypeError(
                "Expected an instance of ABLModel, but received type: {}".format(type(model))
            )
        if not isinstance(reasoner, Reasoner):
            raise TypeError(
                "Expected an instance of Reasoner, but received type: {}".format(type(reasoner))
            )

        self.model = model
        self.reasoner = reasoner

    @abstractmethod
    def predict(self, data_samples: ListData) -> Tuple[List[List[Any]], List[List[Any]]]:
        """Placeholder for predict labels from input."""

    @abstractmethod
    def abduce_pseudo_label(self, data_samples: ListData) -> List[List[Any]]:
        """Placeholder for abduce pseudo labels."""

    @abstractmethod
    def idx_to_pseudo_label(self, data_samples: ListData) -> List[List[Any]]:
        """Placeholder for map label space to symbol space."""

    @abstractmethod
    def pseudo_label_to_idx(self, data_samples: ListData) -> List[List[Any]]:
        """Placeholder for map symbol space to label space."""

    @abstractmethod
    def train(self, train_data: Union[ListData, DataSet]):
        """Placeholder for train loop of ABductive Learning."""

    @abstractmethod
    def valid(self, valid_data: Union[ListData, DataSet]) -> None:
        """Placeholder for model test."""

    @abstractmethod
    def test(self, test_data: Union[ListData, DataSet]) -> None:
        """Placeholder for model validation."""
