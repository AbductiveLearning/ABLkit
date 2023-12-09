from typing import Optional

from ..reasoning import KBBase
from ..structures import ListData
from .base_metric import BaseMetric


class SemanticsMetric(BaseMetric):
    def __init__(self, kb: KBBase = None, prefix: Optional[str] = None) -> None:
        super().__init__(prefix)
        self.kb = kb

    def process(self, data_samples: ListData) -> None:
        pred_pseudo_label_list = data_samples.pred_pseudo_label
        y_list = data_samples.Y
        for pred_pseudo_label, y in zip(pred_pseudo_label_list, y_list):
            if self.kb._check_equal(self.kb.logic_forward(pred_pseudo_label), y):
                self.results.append(1)
            else:
                self.results.append(0)

    def compute_metrics(self) -> dict:
        results = self.results
        metrics = dict()
        metrics["semantics_accuracy"] = sum(results) / len(results)
        return metrics
