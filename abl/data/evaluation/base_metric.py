import logging
from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional

from ...utils import print_log
from ..structures import ListData


class BaseMetric(metaclass=ABCMeta):
    """
    Base class for a metrics.

    The metrics first processes each batch of data_examples and appends the processed
    results to the results list. Then, it computes the metrics of the entire dataset.

    Parameters
    ----------
    prefix : str, optional
        The prefix that will be added in the metrics names to disambiguate homonymous
        metrics of different tasks. If prefix is not provided in the argument,
        self.default_prefix will be used instead. Default to None.

    """

    def __init__(
        self,
        prefix: Optional[str] = None,
    ) -> None:
        self.default_prefix = ""
        self.results: List[Any] = []
        self.prefix = prefix or self.default_prefix

    @abstractmethod
    def process(self, data_examples: ListData) -> None:
        """
        Process one batch of data examples. The processed results should be stored
        in ``self.results``, which will be used to compute the metrics when all
        batches have been processed.

        Parameters
        ----------
        data_examples : ListData
            A batch of data examples.
        """

    @abstractmethod
    def compute_metrics(self) -> dict:
        """
        Compute the metrics from processed results.

        Returns
        -------
        dict
            The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

    def evaluate(self) -> dict:
        """
        Evaluate the model performance of the whole dataset after processing
        all batches.

        Returns
        -------
        dict
            Evaluation metrics dict on the val dataset. The keys are the
            names of the metrics, and the values are corresponding results.
        """
        if len(self.results) == 0:
            print_log(
                f"{self.__class__.__name__} got empty `self.results`. Please "
                "ensure that the processed results are properly added into "
                "`self.results` in `process` method.",
                logger="current",
                level=logging.WARNING,
            )

        metrics = self.compute_metrics()
        # Add prefix to metrics names
        if self.prefix:
            metrics = {"/".join((self.prefix, k)): v for k, v in metrics.items()}

        # reset the results list
        self.results.clear()
        return metrics
