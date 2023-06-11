from ..learning import ABLModel
from ..reasoning import ReasonerBase
from ..evaluation import BaseMetric
from .base_bridge import BaseBridge
from typing import List, Union, Any, Tuple, Dict, Optional
from numpy import ndarray

from torch.utils.data import DataLoader
from ..dataset import BridgeDataset
from ..utils.logger import print_log


class SimpleBridge(BaseBridge):
    def __init__(
        self,
        model: ABLModel,
        abducer: ReasonerBase,
        metric_list: BaseMetric,
    ) -> None:
        super().__init__(model, abducer)
        self.metric_list = metric_list

    def predict(self, X) -> Tuple[List[List[Any]], ndarray]:
        pred_res = self.model.predict(X)
        pred_label, pred_prob = pred_res["label"], pred_res["prob"]
        return pred_label, pred_prob
    
    def abduce_pseudo_label(
        self,
        pred_label: List[List[Any]],
        pred_prob: ndarray,
        pseudo_label: List[List[Any]],
        Y: List[List[Any]],
        max_revision: int = -1,
        require_more_revision: int = 0,
    ) -> List[List[Any]]:
        return self.abducer.batch_abduce(pred_label, pred_prob, pseudo_label, Y, max_revision, require_more_revision)

    def label_to_pseudo_label(
        self, label: List[List[Any]], mapping: Dict = None
    ) -> List[List[Any]]:
        if mapping is None:
            mapping = self.abducer.mapping
        return [[mapping[_label] for _label in sub_list] for sub_list in label]

    def pseudo_label_to_label(
        self, pseudo_label: List[List[Any]], mapping: Dict = None
    ) -> List[List[Any]]:
        if mapping is None:
            mapping = self.abducer.remapping
        return [
            [mapping[_pseudo_label] for _pseudo_label in sub_list]
            for sub_list in pseudo_label
        ]

    def train(
        self,
        train_data: Tuple[List[List[Any]], Optional[List[List[Any]]], List[List[Any]]],
        epochs: int = 50,
        batch_size: Union[int, float] = -1,
        eval_interval: int = 1,
    ):
        dataset = BridgeDataset(*train_data)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=lambda data_list: [list(data) for data in zip(*data_list)],
        )

        for epoch in range(epochs):
            for seg_idx, (X, Z, Y) in enumerate(data_loader):
                pred_label, pred_prob = self.predict(X)
                pred_pseudo_label = self.label_to_pseudo_label(pred_label)
                abduced_pseudo_label = self.abduce_pseudo_label(
                    pred_label, pred_prob, pred_pseudo_label, Y
                )
                abduced_label = self.pseudo_label_to_label(abduced_pseudo_label)
                min_loss = self.model.train(X, abduced_label)

                print_log(
                    f"Epoch(train) [{epoch + 1}] [{(seg_idx + 1):3}/{len(data_loader)}] minimal_loss is {min_loss:.5f}",
                    logger="current",
                )

            if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
                print_log(f"Evaluation start: Epoch(val) [{epoch}]", logger="current")
                self.valid(train_data)

    def _valid(self, data_loader):
        for X, Z, Y in data_loader:
            pred_label, pred_prob = self.predict(X)
            pred_pseudo_label = self.label_to_pseudo_label(pred_label)
            data_samples = dict(
                pred_label=pred_label,
                pred_prob=pred_prob,
                pred_pseudo_label=pred_pseudo_label,
                gt_pseudo_label=Z,
                Y=Y,
                logic_forward=self.abducer.kb.logic_forward,
            )
            for metric in self.metric_list:
                metric.process(data_samples)

        res = dict()
        for metric in self.metric_list:
            res.update(metric.evaluate())
        msg = "Evaluation ended, "
        for k, v in res.items():
            msg += k + f": {v:.3f} "
        print_log(msg, logger="current")

    def valid(self, valid_data, batch_size=1000):
        dataset = BridgeDataset(*valid_data)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=lambda data_list: [list(data) for data in zip(*data_list)],
        )
        self._valid(data_loader)

    def test(self, test_data, batch_size=1000):
        self.valid(test_data, batch_size)
