from typing import Optional

from ablkit.data.evaluation.base_metric import BaseMetric
from ablkit.data.structures import ListData
from ablkit.reasoning import KBBase


class ConsistencyMetric(BaseMetric):
    def __init__(self, kb: KBBase, prefix: Optional[str] = None) -> None:
        super().__init__(prefix)
        self.kb = kb

    def process(self, data_examples: ListData) -> None:
        pred_pseudo_label = data_examples.pred_pseudo_label
        learned_rules = self.kb.learned_rules
        consistent_num = sum(
            [
                self.kb.consist_rule(instance, learned_rules[len(instance)])
                for instance in pred_pseudo_label
            ]
        )
        self.results.append((consistent_num, len(pred_pseudo_label)))

    def compute_metrics(self) -> dict:
        results = self.results
        metrics = dict()
        metrics["consistency"] = sum(t[0] for t in results) / sum(t[1] for t in results)
        return metrics
