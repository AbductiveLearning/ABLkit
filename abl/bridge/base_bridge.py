from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional, Tuple, Union

from ..data.structures import ListData
from ..learning import ABLModel
from ..reasoning import Reasoner


class BaseBridge(metaclass=ABCMeta):
    """
    A base class for bridging learning and reasoning parts.

    This class provides necessary methods that need to be overridden in subclasses
    to construct a typical pipeline of Abductive Learning (corresponding to ``train``),
    which involves the following four methods:

        - predict: Predict class indices on the given data examples.
        - idx_to_pseudo_label: Map indices into pseudo-labels.
        - abduce_pseudo_label: Revise pseudo-labels based on abdutive reasoning.
        - pseudo_label_to_idx: Map revised pseudo-labels back into indices.

    Parameters
    ----------
    model : ABLModel
        The machine learning model wrapped in ``ABLModel``, which is mainly used for
        prediction and model training.
    reasoner : Reasoner
        The reasoning part wrapped in ``Reasoner``, which is used for pseudo-label revision.
    """

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
    def predict(self, data_examples: ListData) -> Tuple[List[List[Any]], List[List[Any]]]:
        """Placeholder for predicting class indices from input."""

    @abstractmethod
    def abduce_pseudo_label(self, data_examples: ListData) -> List[List[Any]]:
        """Placeholder for revising pseudo-labels based on abdutive reasoning."""

    @abstractmethod
    def idx_to_pseudo_label(self, data_examples: ListData) -> List[List[Any]]:
        """Placeholder for mapping indices to pseudo-labels."""

    @abstractmethod
    def pseudo_label_to_idx(self, data_examples: ListData) -> List[List[Any]]:
        """Placeholder for mapping pseudo-labels to indices."""

    def filter_pseudo_label(self, data_examples: ListData) -> List[List[Any]]:
        """Default filter function for pseudo-label."""
        non_empty_idx = [
            i
            for i in range(len(data_examples.abduced_pseudo_label))
            if data_examples.abduced_pseudo_label[i]
        ]
        data_examples.update(data_examples[non_empty_idx])
        return data_examples

    @abstractmethod
    def train(
        self,
        train_data: Union[ListData, Tuple[List[List[Any]], Optional[List[List[Any]]], List[Any]]],
    ):
        """Placeholder for training loop of ABductive Learning."""

    @abstractmethod
    def valid(
        self,
        val_data: Union[ListData, Tuple[List[List[Any]], Optional[List[List[Any]]], List[Any]]],
    ) -> None:
        """Placeholder for model test."""

    @abstractmethod
    def test(
        self,
        test_data: Union[ListData, Tuple[List[List[Any]], Optional[List[List[Any]]], List[Any]]],
    ) -> None:
        """Placeholder for model validation."""
