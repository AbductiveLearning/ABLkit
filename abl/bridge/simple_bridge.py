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
    """
    A basic implementation for bridging machine learning and reasoning parts.

    This class implements the typical pipeline of Abductive learning, which involves
    the following five steps:

        - Predict class probabilities and indices for the given data samples.
        - Map indices into pseudo labels.
        - Revise pseudo labels based on abdutive reasoning.
        - Map the revised pseudo labels to indices.
        - Train the model.

    Parameters
    ----------
    model : ABLModel
        The machine learning model wrapped in ``ABLModel``, which is mainly used for
        prediction and model training.
    reasoner : Reasoner
        The reasoning part wrapped in ``Reasoner``, which is used for pseudo label revision.
    metric_list : List[BaseMetric]
        A list of metrics used for evaluating the model's performance.
    """

    def __init__(
        self,
        model: ABLModel,
        reasoner: Reasoner,
        metric_list: List[BaseMetric],
    ) -> None:
        super().__init__(model, reasoner)
        self.metric_list = metric_list

    def predict(self, data_samples: ListData) -> Tuple[List[ndarray], List[ndarray]]:
        """
        Predict class indices and probabilities (if ``predict_proba`` is implemented in
        ``self.model.base_model``) on the given data samples.

        Parameters
        ----------
        data_samples : ListData
            Data samples on which predictions are to be made.

        Returns
        -------
        Tuple[List[ndarray], List[ndarray]]
            A tuple containing lists of predicted indices and probabilities.
        """
        self.model.predict(data_samples)
        return data_samples.pred_idx, data_samples.pred_prob

    def abduce_pseudo_label(self, data_samples: ListData) -> List[List[Any]]:
        """
        Revise predicted pseudo labels of the given data samples using abduction.

        Parameters
        ----------
        data_samples : ListData
            Data samples containing predicted pseudo labels.

        Returns
        -------
        List[List[Any]]
            A list of abduced pseudo labels for the given data samples.
        """
        self.reasoner.batch_abduce(data_samples)
        return data_samples.abduced_pseudo_label

    def idx_to_pseudo_label(self, data_samples: ListData) -> List[List[Any]]:
        """
        Map indices of data samples into pseudo labels.

        Parameters
        ----------
        data_samples : ListData
            Data samples containing the indices.

        Returns
        -------
        List[List[Any]]
            A list of pseudo labels converted from indices.
        """
        pred_idx = data_samples.pred_idx
        data_samples.pred_pseudo_label = [
            [self.reasoner.mapping[_idx] for _idx in sub_list] for sub_list in pred_idx
        ]
        return data_samples.pred_pseudo_label

    def pseudo_label_to_idx(self, data_samples: ListData) -> List[List[Any]]:
        """
        Map pseudo labels of data samples into indices.

        Parameters
        ----------
        data_samples : ListData
            Data samples containing pseudo labels.

        Returns
        -------
        List[List[Any]]
            A list of indices converted from pseudo labels.
        """
        abduced_idx = [
            [self.reasoner.remapping[_abduced_pseudo_label] for _abduced_pseudo_label in sub_list]
            for sub_list in data_samples.abduced_pseudo_label
        ]
        data_samples.abduced_idx = abduced_idx
        return data_samples.abduced_idx

    def data_preprocess(
        self,
        prefix: str,
        data: Union[ListData, Tuple[List[List[Any]], Optional[List[List[Any]]], List[Any]]],
    ) -> ListData:
        """
        Transform data in the form of (X, gt_pseudo_label, Y) into ListData.

        Parameters
        ----------
        prefix : str
            A prefix indicating the type of data processing (e.g., 'train', 'test').
        data : Union[ListData, Tuple[List[List[Any]], Optional[List[List[Any]]], List[Any]]]
            Data to be preprocessed. Can be ListData or a tuple of lists.

        Returns
        -------
        ListData
            The preprocessed ListData object.
        """
        if isinstance(data, ListData):
            data_samples = data
            if not (
                hasattr(data_samples, "X")
                and hasattr(data_samples, "gt_pseudo_label")
                and hasattr(data_samples, "Y")
            ):
                raise ValueError(
                    f"{prefix}data should have X, gt_pseudo_label and Y attribute but "
                    f"only {data_samples.all_keys()} are provided."
                )
        else:
            X, gt_pseudo_label, Y = data
            data_samples = ListData(X=X, gt_pseudo_label=gt_pseudo_label, Y=Y)

        return data_samples

    def concat_data_samples(
        self, unlabel_data_samples: ListData, label_data_samples: Optional[ListData]
    ) -> ListData:
        """
        Concatenate unlabeled and labeled data samples. ``abduced_pseudo_label`` of unlabeled data samples and ``gt_pseudo_label`` of labeled data samples will be used to train the model.

        Parameters
        ----------
        unlabel_data_samples : ListData
            Unlabeled data samples to concatenate.
        label_data_samples : Optional[ListData]
            Labeled data samples to concatenate, if available.

        Returns
        -------
        ListData
            Concatenated data samples.
        """
        if label_data_samples is None:
            return unlabel_data_samples

        unlabel_data_samples.X = unlabel_data_samples.X + label_data_samples.X
        unlabel_data_samples.abduced_pseudo_label = (
            unlabel_data_samples.abduced_pseudo_label + label_data_samples.gt_pseudo_label
        )
        unlabel_data_samples.Y = unlabel_data_samples.Y + label_data_samples.Y
        return unlabel_data_samples

    def train(
        self,
        train_data: Union[ListData, Tuple[List[List[Any]], Optional[List[List[Any]]], List[Any]]],
        label_data: Optional[
            Union[ListData, Tuple[List[List[Any]], Optional[List[List[Any]]], List[Any]]]
        ] = None,
        val_data: Optional[
            Union[ListData, Tuple[List[List[Any]], Optional[List[List[Any]]], List[Any]]]
        ] = None,
        loops: int = 50,
        segment_size: Union[int, float] = 1.0,
        eval_interval: int = 1,
        save_interval: Optional[int] = None,
        save_dir: Optional[str] = None,
    ):
        """
        A typical training pipeline of Abuductive Learning.

        Parameters
        ----------
        train_data : Union[ListData, Tuple[List[List[Any]], Optional[List[List[Any]]], List[Any]]]
            Training data.
        label_data : Optional[Union[ListData, Tuple[List[List[Any]], Optional[List[List[Any]]], List[Any]]]]
            Data with ``gt_pseudo_label`` that can be used to train the model, by
            default None.
        val_data : Optional[Union[ListData, Tuple[List[List[Any]], Optional[List[List[Any]]], List[Any]]]]
            Validation data, by default None.
        loops : int
            Machine Learning part and Reasoning part will be iteratively optimized
            for ``loops`` times, by default 50.
        segment_size : Union[int, float]
            Data will be split into segments of this size and data in each segment
            will be used together to train the model, by default 1.0.
        eval_interval : int
            The model will be evaluated every ``eval_interval`` loops during training,
            by default 1.
        save_interval : Optional[int]
            The model will be saved every ``eval_interval`` loops during training, by
            default None.
        save_dir : Optional[str]
            Directory to save the model, by default None.
        """
        data_samples = self.data_preprocess("train", train_data)

        if label_data is not None:
            label_data_samples = self.data_preprocess("label", label_data)
        else:
            label_data_samples = None

        if val_data is not None:
            val_data_samples = self.data_preprocess("val", val_data)
        else:
            val_data_samples = data_samples

        if isinstance(segment_size, int):
            if segment_size <= 0:
                raise ValueError("segment_size should be positive.")
        elif isinstance(segment_size, float):
            if 0 < segment_size <= 1:
                segment_size = int(segment_size * len(data_samples))
            else:
                raise ValueError("segment_size should be in (0, 1].")
        else:
            raise ValueError("segment_size should be int or float.")

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
                self.concat_data_samples(sub_data_samples, label_data_samples)
                self.pseudo_label_to_idx(sub_data_samples)
                self.model.train(sub_data_samples)

            if (loop + 1) % eval_interval == 0 or loop == loops - 1:
                print_log(f"Evaluation start: loop(val) [{loop + 1}]", logger="current")
                self._valid(val_data_samples)

            if save_interval is not None and ((loop + 1) % save_interval == 0 or loop == loops - 1):
                print_log(f"Saving model: loop(save) [{loop + 1}]", logger="current")
                self.model.save(
                    save_path=osp.join(save_dir, f"model_checkpoint_loop_{loop + 1}.pth")
                )

    def _valid(self, data_samples: ListData) -> None:
        """
        Internal method for validating the model with given data samples.

        Parameters
        ----------
        data_samples : ListData
            Data samples to be used for validation.
        """
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
        val_data: Union[ListData, Tuple[List[List[Any]], Optional[List[List[Any]]], List[Any]]],
    ) -> None:
        """
        Validate the model with the given validation data.

        Parameters
        ----------
        val_data : Union[ListData, Tuple[List[List[Any]], Optional[List[List[Any]]], List[Any]]]
            Validation data to be used for model evaluation.
        """
        val_data_samples = self.data_preprocess(val_data)
        self._valid(val_data_samples)

    def test(
        self,
        test_data: Union[ListData, Tuple[List[List[Any]], Optional[List[List[Any]]], List[Any]]],
    ) -> None:
        """
        Test the model with the given test data.

        Parameters
        ----------
        test_data : Union[ListData, Tuple[List[List[Any]], Optional[List[List[Any]]], List[Any]]]
            Test data to be used for model evaluation.
        """
        test_data_samples = self.data_preprocess("test", test_data)
        self._valid(test_data_samples)
