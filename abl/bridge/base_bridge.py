from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional, Tuple, Union

from ..learning import ABLModel
from ..reasoning import ReasonerBase
from ..structures import ListData

DataSet = Tuple[List[List[Any]], Optional[List[List[Any]]], List[List[Any]]]


class BaseBridge(metaclass=ABCMeta):
    def __init__(self, model: ABLModel, abducer: ReasonerBase) -> None:
        if not isinstance(model, ABLModel):
            raise TypeError(
                "Expected an instance of ABLModel, but received type: {}".format(
                    type(model)
                )
            )
        if not isinstance(abducer, ReasonerBase):
            raise TypeError(
                "Expected an instance of ReasonerBase, but received type: {}".format(
                    type(abducer)
                )
            )

        self.model = model
        self.abducer = abducer

    @abstractmethod
    def predict(
        self, data_samples: ListData
    ) -> Tuple[List[List[Any]], List[List[Any]]]:
        """Placeholder for predict labels from input."""
        pass

    @abstractmethod
    def abduce_pseudo_label(self, data_samples: ListData) -> List[List[Any]]:
        """Placeholder for abduce pseudo labels."""
        pass

    @abstractmethod
    def idx_to_pseudo_label(self, data_samples: ListData) -> List[List[Any]]:
        """Placeholder for map label space to symbol space."""
        pass

    @abstractmethod
    def pseudo_label_to_idx(self, data_samples: ListData) -> List[List[Any]]:
        """Placeholder for map symbol space to label space."""
        pass

    @abstractmethod
    def train(self, train_data: Union[ListData, DataSet]):
        """Placeholder for train loop of ABductive Learning."""
        pass

    @abstractmethod
    def valid(self, valid_data: Union[ListData, DataSet]) -> None:
        """Placeholder for model test."""
        pass

    @abstractmethod
    def test(self, test_data: Union[ListData, DataSet]) -> None:
        """Placeholder for model validation."""
        pass
