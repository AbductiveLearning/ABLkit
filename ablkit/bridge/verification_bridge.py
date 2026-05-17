"""
Bridge for Verification Learning.

:class:`VerificationBridge` replaces the single-candidate abduction step of
:class:`SimpleBridge` with a top-K enumeration provided by
:class:`~ablkit.reasoning.VerificationReasoner`. For each
segment the bridge trains the model once per top-K candidate, exposing
the model to every assignment that is consistent with the knowledge base.

Reference: https://github.com/VerificationLearning/VerificationLearning
"""

from typing import Any, List, Optional, Tuple, Union

from ..data.evaluation import BaseMetric
from ..data.structures import ListData
from ..learning import ABLModel
from ..reasoning.reasoner import VerificationReasoner
from ..utils import print_log
from .simple_bridge import SimpleBridge


class VerificationBridge(SimpleBridge):
    """
    Bridge implementing the Verification Learning training loop.

    Parameters
    ----------
    model : ABLModel
        Wrapped learning model.
    reasoner : VerificationReasoner
        Top-K reasoner. The bridge reads ``reasoner.top_k`` to decide how
        many training passes to run per segment.
    metric_list : List[BaseMetric]
        Evaluation metrics, identical to :class:`SimpleBridge`.
    """

    def __init__(
        self,
        model: ABLModel,
        reasoner: VerificationReasoner,
        metric_list: List[BaseMetric],
    ) -> None:
        if not isinstance(reasoner, VerificationReasoner):
            raise TypeError(
                "VerificationBridge requires a VerificationReasoner; "
                f"got {type(reasoner).__name__}."
            )
        super().__init__(model, reasoner, metric_list)

    def train(
        self,
        train_data: Union[ListData, Tuple[List[List[Any]], Optional[List[List[Any]]], List[Any]]],
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
    ) -> None:
        """
        Verification Learning training loop. For each segment we predict
        once, enumerate the top-K consistent candidates, then run a
        ``model.train`` pass per candidate.
        """
        data_examples = self.data_preprocess("train", train_data)
        val_data_examples = (
            self.data_preprocess("val", val_data) if val_data is not None else data_examples
        )

        segment_size = self._resolve_segment_size(segment_size, len(data_examples))
        num_segments = (len(data_examples) - 1) // segment_size + 1

        for loop in range(loops):
            for seg_idx in range(num_segments):
                print_log(
                    f"loop(train) [{loop + 1}/{loops}] segment(train) "
                    f"[{seg_idx + 1}/{num_segments}] ",
                    logger="current",
                )
                sub_data_examples = data_examples[
                    seg_idx * segment_size : (seg_idx + 1) * segment_size
                ]
                self._train_one_segment_verification(sub_data_examples)

            self._maybe_eval(val_data_examples, loop, loops, eval_interval)
            self._maybe_save(loop, loops, save_interval, save_dir)

    def _train_one_segment_verification(self, sub_data_examples: ListData) -> None:
        """
        Predict, enumerate top-K candidates, then train once per candidate.
        Each example's k-th training pass uses its k-th candidate (or, if
        the example yielded fewer than k candidates, its last available
        candidate, repeated).
        """
        self.predict(sub_data_examples)
        self.idx_to_pseudo_label(sub_data_examples)
        per_example_candidates = self.reasoner.batch_top_k(sub_data_examples)

        if not per_example_candidates:
            return

        max_k = max(len(cands) for cands in per_example_candidates)
        for k_idx in range(max_k):
            sub_data_examples.abduced_pseudo_label = [
                cands[min(k_idx, len(cands) - 1)] for cands in per_example_candidates
            ]
            self.filter_pseudo_label(sub_data_examples)
            self.pseudo_label_to_idx(sub_data_examples)
            if len(sub_data_examples) == 0:
                continue
            self.model.train(sub_data_examples)
