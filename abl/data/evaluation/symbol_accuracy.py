from typing import Optional

import numpy as np

from ..structures import ListData
from .base_metric import BaseMetric


class SymbolAccuracy(BaseMetric):
    """
    A metrics class for evaluating symbol-level accuracy.

    This class is designed to assess the accuracy of symbol prediction. Symbol accuracy
    are calculated by comparing predicted presudo labels and their ground truth.

    Parameters
    ----------
    prefix : str, optional
        The prefix that will be added to the metrics names to disambiguate homonymous
        metrics of different tasks. Inherits from BaseMetric. Default to None.
    """

    def __init__(self, prefix: Optional[str] = None) -> None:
        super().__init__(prefix)

    def process(self, data_examples: ListData) -> None:
        """
        Processes a batch of data examples.

        This method takes in a batch of data examples, each containing a list of predicted
        pseudo-labels (pred_pseudo_label) and their ground truth (gt_pseudo_label). It
        calculates the accuracy by comparing the two lists. Then, a tuple of correct symbol
        count and total symbol count is appended to `self.results`.

        Parameters
        ----------
        data_examples : ListData
            A batch of data examples, each containing:
            - `pred_pseudo_label`: List of predicted pseudo-labels.
            - `gt_pseudo_label`: List of ground truth pseudo-labels.

        Raises
        ------
        ValueError
            If the lengths of predicted and ground truth symbol lists are not equal.
        """
        pred_pseudo_label_list = data_examples.flatten("pred_pseudo_label")
        gt_pseudo_label_list = data_examples.flatten("gt_pseudo_label")

        if not len(pred_pseudo_label_list) == len(gt_pseudo_label_list):
            raise ValueError("lengthes of pred_pseudo_label and gt_pseudo_label should be equal")

        correct_num = np.sum(np.array(pred_pseudo_label_list) == np.array(gt_pseudo_label_list))

        self.results.append((correct_num, len(pred_pseudo_label_list)))

    def compute_metrics(self) -> dict:
        """
        Compute the symbol accuracy metrics from ``self.results``. It calculates the
        percentage of correctly predicted pseudo-labels over all pseudo-labels.

        Returns
        -------
        dict
            A dictionary containing the computed metrics. It includes the key
            'character_accuracy' which maps to the calculated symbol-level accuracy,
            represented as a float between 0 and 1.

        """
        results = self.results
        metrics = dict()
        metrics["character_accuracy"] = sum(t[0] for t in results) / sum(t[1] for t in results)
        return metrics
