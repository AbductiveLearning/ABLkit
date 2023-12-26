import os.path as osp
from typing import Any, List, Optional, Tuple, Union

from numpy import ndarray

from ..data.evaluation import BaseMetric
from ..data.structures import ListData
from ..learning import ABLModel
from ..reasoning import Reasoner
from ..utils import print_log
from .base_bridge import BaseBridge


class SimpleBridge(BaseBridge):
    """
    A basic implementation for bridging machine learning and reasoning parts.

    This class implements the typical pipeline of Abductive Learning, which involves
    the following five steps:

        - Predict class probabilities and indices for the given data examples.
        - Map indices into pseudo-labels.
        - Revise pseudo-labels based on abdutive reasoning.
        - Map the revised pseudo-labels to indices.
        - Train the model.

    Parameters
    ----------
    model : ABLModel
        The machine learning model wrapped in ``ABLModel``, which is mainly used for
        prediction and model training.
    reasoner : Reasoner
        The reasoning part wrapped in ``Reasoner``, which is used for pseudo-label revision.
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

    def predict(self, data_examples: ListData) -> Tuple[List[ndarray], List[ndarray]]:
        """
        Predict class indices and probabilities (if ``predict_proba`` is implemented in
        ``self.model.base_model``) on the given data examples.

        Parameters
        ----------
        data_examples : ListData
            Data examples on which predictions are to be made.

        Returns
        -------
        Tuple[List[ndarray], List[ndarray]]
            A tuple containing lists of predicted indices and probabilities.
        """
        self.model.predict(data_examples)
        return data_examples.pred_idx, data_examples.pred_prob

    def abduce_pseudo_label(self, data_examples: ListData) -> List[List[Any]]:
        """
        Revise predicted pseudo-labels of the given data examples using abduction.

        Parameters
        ----------
        data_examples : ListData
            Data examples containing predicted pseudo-labels.

        Returns
        -------
        List[List[Any]]
            A list of abduced pseudo-labels for the given data examples.
        """
        self.reasoner.batch_abduce(data_examples)
        return data_examples.abduced_pseudo_label

    def idx_to_pseudo_label(self, data_examples: ListData) -> List[List[Any]]:
        """
        Map indices of data examples into pseudo-labels.

        Parameters
        ----------
        data_examples : ListData
            Data examples containing the indices.

        Returns
        -------
        List[List[Any]]
            A list of pseudo-labels converted from indices.
        """
        pred_idx = data_examples.pred_idx
        data_examples.pred_pseudo_label = [
            [self.reasoner.idx_to_label[_idx] for _idx in sub_list] for sub_list in pred_idx
        ]
        return data_examples.pred_pseudo_label

    def pseudo_label_to_idx(self, data_examples: ListData) -> List[List[Any]]:
        """
        Map pseudo-labels of data examples into indices.

        Parameters
        ----------
        data_examples : ListData
            Data examples containing pseudo-labels.

        Returns
        -------
        List[List[Any]]
            A list of indices converted from pseudo-labels.
        """
        abduced_idx = [
            [
                self.reasoner.label_to_idx[_abduced_pseudo_label]
                for _abduced_pseudo_label in sub_list
            ]
            for sub_list in data_examples.abduced_pseudo_label
        ]
        data_examples.abduced_idx = abduced_idx
        return data_examples.abduced_idx

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
            data_examples = data
            if not (
                hasattr(data_examples, "X")
                and hasattr(data_examples, "gt_pseudo_label")
                and hasattr(data_examples, "Y")
            ):
                raise ValueError(
                    f"{prefix}data should have X, gt_pseudo_label and Y attribute but "
                    f"only {data_examples.all_keys()} are provided."
                )
        else:
            X, gt_pseudo_label, Y = data
            data_examples = ListData(X=X, gt_pseudo_label=gt_pseudo_label, Y=Y)

        return data_examples

    def concat_data_examples(
        self, unlabel_data_examples: ListData, label_data_examples: Optional[ListData]
    ) -> ListData:
        """
        Concatenate unlabeled and labeled data examples. ``abduced_pseudo_label`` of unlabeled data examples and ``gt_pseudo_label`` of labeled data examples will be used to train the model.

        Parameters
        ----------
        unlabel_data_examples : ListData
            Unlabeled data examples to concatenate.
        label_data_examples : Optional[ListData]
            Labeled data examples to concatenate, if available.

        Returns
        -------
        ListData
            Concatenated data examples.
        """
        if label_data_examples is None:
            return unlabel_data_examples

        unlabel_data_examples.X = unlabel_data_examples.X + label_data_examples.X
        unlabel_data_examples.abduced_pseudo_label = (
            unlabel_data_examples.abduced_pseudo_label + label_data_examples.gt_pseudo_label
        )
        unlabel_data_examples.Y = unlabel_data_examples.Y + label_data_examples.Y
        return unlabel_data_examples

    def train(
        self,
        train_data: Union[ListData, Tuple[List[List[Any]], Optional[List[List[Any]]], List[Any]]],
        label_data: Optional[
            Union[ListData, Tuple[List[List[Any]], List[List[Any]], List[Any]]]
        ] = None,
        val_data: Optional[
            Union[ListData, Tuple[List[List[Any]], Optional[List[List[Any]]], Optional[List[Any]]]]
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
            Training data should be in the form of ``(X, gt_pseudo_label, Y)`` or a ``ListData``
            object with ``X``, ``gt_pseudo_label`` and ``Y`` attributes.
            - ``X`` is a list of sublists representing the input data.
            - ``gt_pseudo_label`` is only used to evaluate the performance of the ``ABLModel`` but not
            to train. ``gt_pseudo_label`` can be ``None``.
            - ``Y`` is a list representing the ground truth reasoning result for each sublist in ``X``.
        label_data : Optional[Union[ListData, Tuple[List[List[Any]], List[List[Any]], List[Any]]]]
            Labeled data should be in the same format as ``train_data``. The only difference is
            that the ``gt_pseudo_label`` in ``label_data`` should not be ``None`` and will be
            utilized to train the model. Defaults to None.
        val_data : Optional[Union[ListData, Tuple[List[List[Any]], Optional[List[List[Any]]], Optional[List[Any]]]]]
            Validation data should be in the same format as ``train_data``. Both ``gt_pseudo_label``
            and ``Y`` can be either None or not, which depends on the evaluation metircs in
            ``self.metric_list``. If ``val_data`` is None, ``train_data`` will be used to validate the
            model during training time. Defaults to None.
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
        data_examples = self.data_preprocess("train", train_data)

        if label_data is not None:
            label_data_examples = self.data_preprocess("label", label_data)
        else:
            label_data_examples = None

        if val_data is not None:
            val_data_examples = self.data_preprocess("val", val_data)
        else:
            val_data_examples = data_examples

        if isinstance(segment_size, int):
            if segment_size <= 0:
                raise ValueError("segment_size should be positive.")
        elif isinstance(segment_size, float):
            if 0 < segment_size <= 1:
                segment_size = int(segment_size * len(data_examples))
            else:
                raise ValueError("segment_size should be in (0, 1].")
        else:
            raise ValueError("segment_size should be int or float.")

        for loop in range(loops):
            for seg_idx in range((len(data_examples) - 1) // segment_size + 1):
                print_log(
                    f"loop(train) [{loop + 1}/{loops}] segment(train) "
                    f"[{(seg_idx + 1)}/{(len(data_examples) - 1) // segment_size + 1}] ",
                    logger="current",
                )

                sub_data_examples = data_examples[
                    seg_idx * segment_size : (seg_idx + 1) * segment_size
                ]
                self.predict(sub_data_examples)
                self.idx_to_pseudo_label(sub_data_examples)
                self.abduce_pseudo_label(sub_data_examples)
                self.filter_pseudo_label(sub_data_examples)
                self.concat_data_examples(sub_data_examples, label_data_examples)
                self.pseudo_label_to_idx(sub_data_examples)
                self.model.train(sub_data_examples)

            if (loop + 1) % eval_interval == 0 or loop == loops - 1:
                print_log(f"Eval start: loop(val) [{loop + 1}]", logger="current")
                self._valid(val_data_examples)

            if save_interval is not None and ((loop + 1) % save_interval == 0 or loop == loops - 1):
                print_log(f"Saving model: loop(save) [{loop + 1}]", logger="current")
                self.model.save(
                    save_path=osp.join(save_dir, f"model_checkpoint_loop_{loop + 1}.pth")
                )

    def _valid(self, data_examples: ListData) -> None:
        """
        Internal method for validating the model with given data examples.

        Parameters
        ----------
        data_examples : ListData
            Data examples to be used for validation.
        """
        self.predict(data_examples)
        self.idx_to_pseudo_label(data_examples)

        for metric in self.metric_list:
            metric.process(data_examples)

        res = dict()
        for metric in self.metric_list:
            res.update(metric.evaluate())
        msg = "Evaluation ended, "
        for k, v in res.items():
            msg += k + f": {v:.3f} "
        print_log(msg, logger="current")

    def valid(
        self,
        val_data: Union[
            ListData, Tuple[List[List[Any]], Optional[List[List[Any]]], Optional[List[Any]]]
        ],
    ) -> None:
        """
        Validate the model with the given validation data.

        Parameters
        ----------
        val_data : Union[ListData, Tuple[List[List[Any]], Optional[List[List[Any]]], Optional[List[Any]]]]
            Validation data should be in the form of ``(X, gt_pseudo_label, Y)`` or a ``ListData`` object
            with ``X``, ``gt_pseudo_label`` and ``Y`` attributes. Both ``gt_pseudo_label`` and ``Y`` can be
            either None or not, which depends on the evaluation metircs in ``self.metric_list``.
        """
        val_data_examples = self.data_preprocess("val", val_data)
        self._valid(val_data_examples)

    def test(
        self,
        test_data: Union[
            ListData, Tuple[List[List[Any]], Optional[List[List[Any]]], Optional[List[Any]]]
        ],
    ) -> None:
        """
        Test the model with the given test data.

        Parameters
        ----------
        test_data : Union[ListData, Tuple[List[List[Any]], Optional[List[List[Any]]], Optional[List[Any]]]]
            Test data should be in the form of ``(X, gt_pseudo_label, Y)`` or a ``ListData`` object with ``X``,
            ``gt_pseudo_label`` and ``Y`` attributes. Both ``gt_pseudo_label`` and ``Y`` can be either None or
            not, which depends on the evaluation metircs in ``self.metric_list``.
        """
        print_log("Test start:", logger="current")
        test_data_examples = self.data_preprocess("test", test_data)
        self._valid(test_data_examples)
