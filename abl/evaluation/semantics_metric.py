from typing import Optional, Sequence

from ..reasoning import KBBase
from .base_metric import BaseMetric


class SemanticsMetric(BaseMetric):
    def __init__(self, kb: KBBase = None, prefix: Optional[str] = None) -> None:
        super().__init__(prefix)
        self.kb = kb

    def process(self, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            if self.kb.check_equal(data_sample, data_sample["Y"][0]):
                self.results.append(1)
            else:
                self.results.append(0)

    def compute_metrics(self, results: list) -> dict:
        metrics = dict()
        metrics["semantics_accuracy"] = sum(results) / len(results)
        return metrics
