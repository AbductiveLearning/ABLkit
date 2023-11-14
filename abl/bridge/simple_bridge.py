import os.path as osp
from typing import Any, Dict, List, Optional, Tuple, Union

from numpy import ndarray

from ..evaluation import BaseMetric
from ..learning import ABLModel
from ..reasoning import ReasonerBase
from ..structures import ListData
from ..utils import print_log
from .base_bridge import BaseBridge, DataSet


class SimpleBridge(BaseBridge):
    def __init__(
        self,
        model: ABLModel,
        abducer: ReasonerBase,
        metric_list: List[BaseMetric],
    ) -> None:
        super().__init__(model, abducer)
        self.metric_list = metric_list

    # TODO: add abducer.mapping to the property of SimpleBridge

    def predict(self, data_samples: ListData) -> Tuple[List[ndarray], List[ndarray]]:
        self.model.predict(data_samples)
        return data_samples["pred_idx"], data_samples.get("pred_prob", None)

    def abduce_pseudo_label(
        self,
        data_samples: ListData,
        max_revision: int = -1,
        require_more_revision: int = 0,
    ) -> List[List[Any]]:
        self.abducer.batch_abduce(data_samples, max_revision, require_more_revision)
        return data_samples["abduced_pseudo_label"]

    def idx_to_pseudo_label(
        self, data_samples: ListData, mapping: Optional[Dict] = None
    ) -> List[List[Any]]:
        if mapping is None:
            mapping = self.abducer.mapping
        pred_idx = data_samples.pred_idx
        data_samples.pred_pseudo_label = [
            [mapping[_idx] for _idx in sub_list] for sub_list in pred_idx
        ]
        return data_samples["pred_pseudo_label"]

    def pseudo_label_to_idx(
        self, data_samples: ListData, mapping: Optional[Dict] = None
    ) -> List[List[Any]]:
        if mapping is None:
            mapping = self.abducer.remapping
        abduced_idx = [
            [mapping[_abduced_pseudo_label] for _abduced_pseudo_label in sub_list]
            for sub_list in data_samples.abduced_pseudo_label
        ]
        data_samples.abduced_idx = abduced_idx
        return data_samples["abduced_idx"]

    def data_preprocess(self, X: List[Any], gt_pseudo_label: List[Any], Y: List[Any]) -> ListData:
        data_samples = ListData()

        data_samples.X = X
        data_samples.gt_pseudo_label = gt_pseudo_label
        data_samples.Y = Y

        return data_samples

    def train(
        self,
        train_data: Union[ListData, DataSet],
        loops: int = 50,
        segment_size: Union[int, float] = -1,
        eval_interval: int = 1,
        save_interval: Optional[int] = None,
        save_dir: Optional[str] = None,
    ):
        if isinstance(train_data, ListData):
            data_samples = train_data
        else:
            data_samples = self.data_preprocess(*train_data)

        for loop in range(loops):
            for seg_idx in range((len(data_samples) - 1) // segment_size + 1):
                sub_data_samples = data_samples[
                    seg_idx * segment_size : (seg_idx + 1) * segment_size
                ]
                self.predict(sub_data_samples)
                self.idx_to_pseudo_label(sub_data_samples)
                self.abduce_pseudo_label(sub_data_samples)
                self.pseudo_label_to_idx(sub_data_samples)
                loss = self.model.train(sub_data_samples)

                print_log(
                    f"loop(train) [{loop + 1}/{loops}] segment(train) [{(seg_idx + 1)}/{(len(data_samples) - 1) // segment_size + 1}] model loss is {loss:.5f}",
                    logger="current",
                )

            if (loop + 1) % eval_interval == 0 or loop == loops - 1:
                print_log(f"Evaluation start: loop(val) [{loop + 1}]", logger="current")
                self.valid(train_data)

            if save_interval is not None and ((loop + 1) % save_interval == 0 or loop == loops - 1):
                print_log(f"Saving model: loop(save) [{loop + 1}]", logger="current")
                self.model.save(save_path=osp.join(save_dir, f"model_checkpoint_loop_{loop + 1}.pth"))

    def _valid(self, data_samples: ListData, batch_size: int = 128) -> None:
        for seg_idx in range((len(data_samples) - 1) // batch_size + 1):
            sub_data_samples = data_samples[seg_idx * batch_size : (seg_idx + 1) * batch_size]
            self.predict(sub_data_samples)
            self.idx_to_pseudo_label(sub_data_samples)

            for metric in self.metric_list:
                metric.process(sub_data_samples)

        res = dict()
        for metric in self.metric_list:
            res.update(metric.evaluate())
        msg = "Evaluation ended, "
        for k, v in res.items():
            msg += k + f": {v:.3f} "
        print_log(msg, logger="current")

    def valid(self, valid_data: Union[ListData, DataSet], batch_size: int = 128) -> None:
        if not isinstance(valid_data, ListData):
            data_samples = self.data_preprocess(*valid_data)
        else:
            data_samples = valid_data
        self._valid(data_samples, batch_size=batch_size)

    def test(self, test_data: Union[ListData, DataSet], batch_size: int = 128) -> None:
        self.valid(test_data, batch_size=batch_size)
