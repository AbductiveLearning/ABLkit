from typing import Optional

from ablkit.reasoning import KBBase
from ablkit.data import BaseMetric, ListData

class BDDReasoningMetric(BaseMetric):
    def __init__(self, kb: KBBase, prefix: Optional[str] = None) -> None:
        super().__init__(prefix)
        self.kb = kb

    def process(self, data_examples: ListData) -> None:
        pred_pseudo_label_list = data_examples.pred_pseudo_label
        y_list = data_examples.Y
        x_list = data_examples.X
        for pred_pseudo_label, y, x in zip(pred_pseudo_label_list, y_list, x_list):
            pred_y = self.kb.logic_forward(pred_pseudo_label, *(x,) if self.kb._num_args == 2 else ())
            for py, yy in zip(pred_y, y):
                self.results.append(int(py == yy))

    def compute_metrics(self) -> dict:
        results = self.results
        metrics = dict()
        metrics["reasoning_accuracy"] = sum(results) / len(results)
        return metrics