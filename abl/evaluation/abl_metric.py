from typing import Optional, Sequence, Callable
from .base_metric import BaseMetric


class ABLMetric(BaseMetric):
    def __init__(self, prefix: Optional[str] = None) -> None:
        super().__init__(prefix)

    def process(self, data_samples: Sequence[dict]) -> None:
        pred_pseudo_label = data_samples["pred_pseudo_label"]
        gt_Y = data_samples["Y"]
        logic_forward = data_samples["logic_forward"]

        for pred_z, y in zip(pred_pseudo_label, gt_Y):
            if logic_forward(pred_z) == y:
                self.results.append(1)
            else:
                self.results.append(0)
    
    def compute_metrics(self, results: list) -> dict:
        metrics = dict()
        metrics["abl_accuracy"] = sum(results) / len(results)
        return metrics