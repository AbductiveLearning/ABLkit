import os.path as osp
from typing import Any, List, Optional, Tuple, Union

from ablkit.bridge import SimpleBridge
from ablkit.data.evaluation.base_metric import BaseMetric
from ablkit.data.structures.list_data import ListData
from ablkit.reasoning import A3BLReasoner
from ablkit.utils import print_log

from models.a3bl_model import A3BLModel

class A3BLBridge(SimpleBridge):
    """
    An ambiguity-aware implementation for bridging machine learning and reasoning parts.

    Involves the following five steps:
        - Predict class probabilities and indices for the given data examples.
        - Map indices into pseudo-labels.
        - Enumerate all valid pseudo-labels.
        - Revise pseudo-labels to label distribution based on the class probabilities.
        - Train the model.

    Parameters
    ----------
    model : A3BLModel
        The machine learning model wrapped in ``A3BLModel``, which is mainly used for
        prediction and model training.
    reasoner : A3BLReasoner
        The reasoning part wrapped in ``A3blReasoner``, which is used for pseudo-label enumeration.
    metric_list : List[BaseMetric]
        A list of metrics used for evaluating the model's performance.
    """

    def __init__(
        self,
        model: A3BLModel,
        reasoner: A3BLReasoner,
        metric_list: List[BaseMetric],
    ):
        super().__init__(model, reasoner, metric_list)

    def abduce_soft_label(self, data_examples: ListData) -> List[List[Any]]:
        """
        Revise predicted pseudo-labels to a soft label, given data examples using abduction.

        Parameters
        ----------
        data_examples : ListData
            Data examples containing predicted pseudo-labels.

        Returns
        -------
        List[List[Any]]
            A list of abduced soft labels for the given data examples.
        """
        self.reasoner.batch_abduce(data_examples)
        return data_examples.abduced_soft_label

    def train_data_iter(
        self,
        train_data,
        val_data = None,
        segment_size= 1.0,
    ):        
        data_examples = self.data_preprocess("train", train_data)

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
        
        for seg_idx in range((len(data_examples) - 1) // segment_size + 1):
            sub_data_examples = data_examples[seg_idx * segment_size : (seg_idx + 1) * segment_size]
            yield sub_data_examples, val_data_examples
        
        

    def train(
        self,
        train_data: Union[
            ListData, Tuple[List[List[Any]], Optional[List[List[Any]]], List[Any]]
        ],
        val_data: Optional[
            Union[
                ListData,
                Tuple[List[List[Any]], Optional[List[List[Any]]], Optional[List[Any]]],
            ]
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
            - ``gt_pseudo_label`` is only used to evaluate the performance of the ``ABLModel`` but
            not to train. ``gt_pseudo_label`` can be ``None``.
            - ``Y`` is a list representing the ground truth reasoning result for each sublist
            in ``X``.
        label_data : Union[ListData, Tuple[List[List[Any]], List[List[Any]], List[Any]]], optional
            Labeled data should be in the same format as ``train_data``. The only difference is
            that the ``gt_pseudo_label`` in ``label_data`` should not be ``None`` and will be
            utilized to train the model. Defaults to None.
        val_data : Union[ListData, Tuple[List[List[Any]], Optional[List[List[Any]]], Optional[List[Any]]]], optional # noqa: E501 pylint: disable=line-too-long
            Validation data should be in the same format as ``train_data``. Both ``gt_pseudo_label``
            and ``Y`` can be either None or not, which depends on the evaluation metircs in
            ``self.metric_list``. If ``val_data`` is None, ``train_data`` will be used to validate
            the model during training time. Defaults to None.
        loops : int
            Learning part and Reasoning part will be iteratively optimized
            for ``loops`` times. Defaults to 50.
        segment_size : Union[int, float]
            Data will be split into segments of this size and data in each segment
            will be used together to train the model. Defaults to 1.0.
        eval_interval : int
            The model will be evaluated every ``eval_interval`` loop during training,
            Defaults to 1.
        save_interval : int, optional
            The model will be saved every ``eval_interval`` loop during training.
            Defaults to None.
        save_dir : str, optional
            Directory to save the model. Defaults to None.
        """


        for loop in range(loops):
            for train_examples_batch, val_examples_batch in self.train_data_iter(train_data, val_data, segment_size):
                print_log(f"loop(train) [{loop + 1}/{loops}] segment(train) ",logger="current")
                self.predict(train_examples_batch)
                self.idx_to_pseudo_label(train_examples_batch)
                self.abduce_pseudo_label(train_examples_batch)
                self.filter_pseudo_label(train_examples_batch)
                self.pseudo_label_to_idx(train_examples_batch)
                self.model.train(train_examples_batch)

            if (loop + 1) % eval_interval == 0 or loop == loops - 1:
                print_log(f"Eval start: loop(val) [{loop + 1}]", logger="current")
                self._valid(val_examples_batch)

            if save_interval is not None and ((loop + 1) % save_interval == 0 or loop == loops - 1):
                print_log(f"Saving model: loop(save) [{loop + 1}]", logger="current")
                self.model.save(save_path=osp.join(save_dir, f"model_checkpoint_loop_{loop + 1}.pth"))
