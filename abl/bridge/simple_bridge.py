import os.path as osp
from typing import Any, List, Optional, Tuple, Union

from numpy import ndarray

from ..evaluation import BaseMetric
from ..learning import ABLModel
from ..reasoning import Reasoner
from ..structures import ListData
from ..utils import print_log
from .base_bridge import BaseBridge


class SimpleBridge(BaseBridge):
    def __init__(
        self,
        model: ABLModel,
        reasoner: Reasoner,
        metric_list: List[BaseMetric],
    ) -> None:
        super().__init__(model, reasoner)
        self.metric_list = metric_list

    def predict(self, data_samples: ListData) -> Tuple[List[ndarray], List[ndarray]]:
        self.model.predict(data_samples)
        return data_samples.pred_idx, data_samples.pred_prob

    def abduce_pseudo_label(self, data_samples: ListData) -> List[List[Any]]:
        self.reasoner.batch_abduce(data_samples)
        return data_samples.abduced_pseudo_label

    def idx_to_pseudo_label(self, data_samples: ListData) -> List[List[Any]]:
        pred_idx = data_samples.pred_idx
        data_samples.pred_pseudo_label = [
            [self.reasoner.mapping[_idx] for _idx in sub_list] for sub_list in pred_idx
        ]
        return data_samples.pred_pseudo_label

    def pseudo_label_to_idx(self, data_samples: ListData) -> List[List[Any]]:
        abduced_idx = [
            [self.reasoner.remapping[_abduced_pseudo_label] for _abduced_pseudo_label in sub_list]
            for sub_list in data_samples.abduced_pseudo_label
        ]
        data_samples.abduced_idx = abduced_idx
        return data_samples.abduced_idx

    def data_preprocess(self, X: List[Any], gt_pseudo_label: List[Any], Y: List[Any]) -> ListData:
        data_samples = ListData()

        data_samples.X = X
        data_samples.gt_pseudo_label = gt_pseudo_label
        data_samples.Y = Y

        return data_samples

    def train(
        self,
        train_data: Union[ListData, Tuple[List[List[Any]], Optional[List[List[Any]]], List[Any]]],
        val_data: Optional[
            Union[ListData, Tuple[List[List[Any]], Optional[List[List[Any]]], List[Any]]]
        ] = None,
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

        if isinstance(segment_size, int) and segment_size == 0:
            raise ValueError("segment_size should be positive.")

        if isinstance(segment_size, float):
            if 0 < segment_size <= 1:
                segment_size = int(segment_size * len(data_samples))
            else:
                raise ValueError("segment_size should be in (0, 1].")

        for loop in range(loops):
            for seg_idx in range((len(data_samples) - 1) // segment_size + 1):
                print_log(
                    f"loop(train) [{loop + 1}/{loops}] segment(train) "
                    f"[{(seg_idx + 1)}/{(len(data_samples) - 1) // segment_size + 1}] ",
                    logger="current",
                )
                
                sub_data_samples = data_samples[
                    seg_idx * segment_size : (seg_idx + 1) * segment_size
                ]
                self.predict(sub_data_samples)
                self.idx_to_pseudo_label(sub_data_samples)
                self.abduce_pseudo_label(sub_data_samples)
                self.filter_pseudo_label(sub_data_samples)
                self.pseudo_label_to_idx(sub_data_samples)
                self.model.train(sub_data_samples)

            if (loop + 1) % eval_interval == 0 or loop == loops - 1:
                print_log(f"Evaluation start: loop(val) [{loop + 1}]", logger="current")
                if val_data is not None:
                    self.valid(val_data)
                else:
                    self.valid(train_data)

            if save_interval is not None and ((loop + 1) % save_interval == 0 or loop == loops - 1):
                print_log(f"Saving model: loop(save) [{loop + 1}]", logger="current")
                self.model.save(
                    save_path=osp.join(save_dir, f"model_checkpoint_loop_{loop + 1}.pth")
                )

    def _valid(self, data_samples: ListData) -> None:
        self.predict(data_samples)
        self.idx_to_pseudo_label(data_samples)

        for metric in self.metric_list:
            metric.process(data_samples)

        res = dict()
        for metric in self.metric_list:
            res.update(metric.evaluate())
        msg = "Evaluation ended, "
        for k, v in res.items():
            msg += k + f": {v:.3f} "
        print_log(msg, logger="current")

    def valid(
        self,
        valid_data: Union[ListData, Tuple[List[List[Any]], Optional[List[List[Any]]], List[Any]]],
    ) -> None:
        if not isinstance(valid_data, ListData):
            data_samples = self.data_preprocess(*valid_data)
        else:
            data_samples = valid_data
        self._valid(data_samples)

    def test(
        self,
        test_data: Union[ListData, Tuple[List[List[Any]], Optional[List[List[Any]]], List[Any]]],
    ) -> None:
        self.valid(test_data)
