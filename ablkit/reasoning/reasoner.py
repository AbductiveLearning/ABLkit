"""
This module contains the class Reasoner, which is used for minimizing the inconsistency
between the knowledge base and learning models.

Copyright (c) 2024 LAMDA.  All rights reserved.
"""

import heapq
import inspect
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

import numpy as np
from zoopt import Dimension, Objective, Opt, Parameter, Solution

from ..data.structures import ListData
from ..reasoning import KBBase
from ..utils.utils import (
    avg_confidence_dist,
    confidence_dist,
    hamming_dist,
    rejection_dist,
    similarity_dist,
)


class Reasoner:
    """
    Reasoner for minimizing the inconsistency between the knowledge base and learning models.

    Parameters
    ----------
    kb : class KBBase
        The knowledge base to be used for reasoning.
    dist_func : Union[str, Callable], optional
        The distance function used to determine the cost list between each
        candidate and the given prediction. The cost is also referred to as a consistency
        measure, wherein the candidate with lowest cost is selected as the final
        abduced label. It can be either a string representing a predefined distance
        function or a callable function. The available predefined distance functions:
        'hamming' | 'confidence' | 'avg_confidence' | 'similarity' | 'rejection'.
        'hamming' directly calculates the Hamming distance between the predicted
        pseudo-label in the data example and each candidate. 'confidence' and
        'avg_confidence' calculate the confidence distance between the predicted
        probabilities and each candidate, defined as ``1 - product`` and
        ``1 - average`` of the candidate's per-symbol probabilities respectively.
        'similarity' compares candidates against the geometry of the model's
        embeddings (requires the base model to expose ``extract_features``;
        ``ABLModel`` then stores the result on ``data_example.embeddings``).
        'rejection' combines confidence distance with a candidate-complexity penalty,
        favoring shorter candidates when scores are close.
        Alternatively, the callable function should have the signature
        ``dist_func(data_example, candidates, candidate_idxs, reasoning_results)`` and must
        return a cost list. Each element in this cost list should be a numerical value
        representing the cost for each candidate, and the list should have the same length
        as candidates. Defaults to 'confidence'.
    idx_to_label : dict, optional
        A mapping from index in the base model to label. If not provided, a default
        order-based index to label mapping is created. Defaults to None.
    max_revision : Union[int, float], optional
        The upper limit on the number of revisions for each data example when
        performing abductive reasoning. If float, denotes the fraction of the total
        length that can be revised. A value of -1 implies no restriction on the
        number of revisions. Defaults to -1.
    require_more_revision : int, optional
        Specifies additional number of revisions permitted beyond the minimum required
        when performing abductive reasoning. Defaults to 0.
    use_zoopt : bool, optional
        Whether to use ZOOpt library during abductive reasoning. Defaults to False.
    """

    def __init__(
        self,
        kb: KBBase,
        dist_func: Union[str, Callable] = "confidence",
        idx_to_label: Optional[dict] = None,
        max_revision: Union[int, float] = -1,
        require_more_revision: int = 0,
        use_zoopt: bool = False,
    ):
        self.kb = kb
        self._check_valid_dist(dist_func)
        self.dist_func = dist_func
        self.use_zoopt = use_zoopt
        self.max_revision = max_revision
        self.require_more_revision = require_more_revision

        if idx_to_label is None:
            self.idx_to_label = {
                index: label for index, label in enumerate(self.kb.pseudo_label_list)
            }
        else:
            self._check_valid_idx_to_label(idx_to_label)
            self.idx_to_label = idx_to_label
        self.label_to_idx = dict(zip(self.idx_to_label.values(), self.idx_to_label.keys()))

    def _check_valid_dist(self, dist_func):
        if isinstance(dist_func, str):
            valid = ["hamming", "confidence", "avg_confidence", "similarity", "rejection"]
            if dist_func not in valid:
                raise NotImplementedError(
                    f"Valid options for predefined dist_func are {valid}, "
                    f"but got {dist_func!r}."
                )
            return
        elif callable(dist_func):
            params = inspect.signature(dist_func).parameters.values()
            if len(params) != 4:
                raise ValueError(
                    "User-defined dist_func must have exactly four parameters, "
                    + f"but got {len(params)}."
                )
            return
        else:
            raise TypeError(
                f"dist_func must be a string or a callable function, but got {type(dist_func)}."
            )

    def _check_valid_idx_to_label(self, idx_to_label):
        if not isinstance(idx_to_label, dict):
            raise TypeError(f"idx_to_label should be dict, but got {type(idx_to_label)}.")
        for key, value in idx_to_label.items():
            if not isinstance(key, int):
                raise ValueError(f"All keys in the idx_to_label must be integers, but got {key}.")
            if value not in self.kb.pseudo_label_list:
                raise ValueError(
                    "All values in the idx_to_label must be in the pseudo_label_list, "
                    + f"but got {value}."
                )

    def _get_one_candidate(
        self,
        data_example: ListData,
        candidates: List[List[Any]],
        reasoning_results: List[Any],
    ) -> List[Any]:
        """
        Due to the nondeterminism of abductive reasoning, there could be multiple candidates
        satisfying the knowledge base. When this happens, return one candidate that has the
        minimum cost. If no candidates are provided, an empty list is returned.

        Parameters
        ----------
        data_example : ListData
            Data example.
        candidates : List[List[Any]]
            Multiple possible candidates.
        reasoning_results : List[Any]
            Corresponding reasoning results of the candidates.

        Returns
        -------
        List[Any]
            A selected candidate.
        """
        if len(candidates) == 0:
            return []
        elif len(candidates) == 1:
            return candidates[0]
        else:
            cost_array = self._get_cost_list(data_example, candidates, reasoning_results)
            candidate = candidates[np.argmin(cost_array)]
            return candidate

    def _get_cost_list(
        self,
        data_example: ListData,
        candidates: List[List[Any]],
        reasoning_results: List[Any],
    ) -> Union[List[Union[int, float]], np.ndarray]:
        """
        Get the list of costs between each candidate and the given data example.

        Parameters
        ----------
        data_example : ListData
            Data example.
        candidates : List[List[Any]]
            Multiple possible candidates.
        reasoning_results : List[Any]
            Corresponding reasoning results of the candidates.

        Returns
        -------
        Union[List[Union[int, float]], np.ndarray]
            The list of costs.
        """
        if self.dist_func == "hamming":
            return hamming_dist(data_example.pred_pseudo_label, candidates)
        elif self.dist_func == "confidence":
            candidates_idxs = [[self.label_to_idx[x] for x in c] for c in candidates]
            return confidence_dist(data_example.pred_prob, candidates_idxs)
        elif self.dist_func == "avg_confidence":
            candidates_idxs = [[self.label_to_idx[x] for x in c] for c in candidates]
            return avg_confidence_dist(data_example.pred_prob, candidates_idxs)
        elif self.dist_func == "similarity":
            embeddings = getattr(data_example, "embeddings", None)
            if embeddings is None:
                raise ValueError(
                    "dist_func='similarity' requires the base model to expose an "
                    "extract_features(X=...) method so ABLModel can populate "
                    "data_example.embeddings."
                )
            candidates_idxs = [[self.label_to_idx[x] for x in c] for c in candidates]
            return similarity_dist(embeddings, candidates_idxs=candidates_idxs)
        elif self.dist_func == "rejection":
            candidates_idxs = [[self.label_to_idx[x] for x in c] for c in candidates]
            return rejection_dist(data_example.pred_prob, candidates_idxs=candidates_idxs)
        else:
            candidates_idxs = [[self.label_to_idx[x] for x in c] for c in candidates]
            cost_list = self.dist_func(data_example, candidates, candidates_idxs, reasoning_results)
            if len(cost_list) != len(candidates):
                raise ValueError(
                    "The length of the array returned by dist_func must be equal to the number "
                    + f"of candidates. Expected length {len(candidates)}, but got {len(cost_list)}."
                )
            return cost_list

    def _zoopt_get_solution(
        self,
        symbol_num: int,
        data_example: ListData,
        max_revision_num: int,
    ) -> Solution:
        """
        Get the optimal solution using ZOOpt library. From the solution, we can get a list of
        boolean values, where '1' (True) indicates the indices chosen to be revised.

        Parameters
        ----------
        symbol_num : int
            Number of total symbols.
        data_example : ListData
            Data example.
        max_revision_num : int
            Specifies the maximum number of revisions allowed.

        Returns
        -------
        Solution
            The solution for ZOOpt library.
        """
        dimension = Dimension(size=symbol_num, regs=[[0, 1]] * symbol_num, tys=[False] * symbol_num)
        objective = Objective(
            lambda sol: self.zoopt_score(symbol_num, data_example, sol),
            dim=dimension,
            constraint=lambda sol: self._constrain_revision_num(sol, max_revision_num),
        )
        parameter = Parameter(
            budget=self.zoopt_budget(symbol_num), intermediate_result=False, autoset=True
        )
        solution = Opt.min(objective, parameter)
        return solution

    def zoopt_score(
        self,
        symbol_num: int,
        data_example: ListData,
        sol: Solution,
    ) -> int:
        """
        Set the score for a solution. A lower score suggests that ZOOpt library
        has a higher preference for this solution.

        Parameters
        ----------
        symbol_num : int
            Number of total symbols.
        data_example : ListData
            Data example.
        sol: Solution
            The solution for ZOOpt library.

        Returns
        -------
        int
            The score for the solution.
        """
        revision_idx = np.where(sol.get_x() != 0)[0]
        candidates, reasoning_results = self.kb.revise_at_idx(
            data_example.pred_pseudo_label, data_example.Y, data_example.X, revision_idx
        )
        if len(candidates) > 0:
            return np.min(self._get_cost_list(data_example, candidates, reasoning_results))
        else:
            return symbol_num

    def zoopt_budget(self, symbol_num: int) -> int:
        """
        Set the budget for ZOOpt optimization. The budget can be dynamic relying on
        the number of symbols considered, e.g., the default implementation shown below.
        Alternatively, it can be a fixed value, such as simply setting it to 100.

        Parameters
        ----------
        symbol_num : int
            The number of symbols to be considered in the ZOOpt optimization process.

        Returns
        -------
        int
            The budget for ZOOpt optimization.
        """
        return 10 * symbol_num

    def _constrain_revision_num(self, solution: Solution, max_revision_num: int) -> int:
        """
        Constrain that the total number of revisions chosen by the solution does not exceed
        maximum number of revisions allowed.
        """
        x = solution.get_x()
        return max_revision_num - x.sum()

    def _get_max_revision_num(self, max_revision: Union[int, float], symbol_num: int) -> int:
        """
        Get the maximum revision number according to input ``max_revision``.
        """
        if not isinstance(max_revision, (int, float)):
            raise TypeError(f"Parameter must be of type int or float, but got {type(max_revision)}")

        if max_revision == -1:
            return symbol_num
        if isinstance(max_revision, float):
            if not 0 <= max_revision <= 1:
                raise ValueError(
                    "If max_revision is a float, it must be between 0 and 1, "
                    + f"but got {max_revision}"
                )
            return round(symbol_num * max_revision)
        if max_revision < 0:
            raise ValueError(
                f"If max_revision is an int, it must be non-negative, but got {max_revision}"
            )
        return max_revision

    def abduce(self, data_example: ListData) -> List[Any]:
        """
        Perform abductive reasoning on the given data example.

        Parameters
        ----------
        data_example : ListData
            Data example.

        Returns
        -------
        List[Any]
            A revised pseudo-labels of the example through abductive reasoning, which is compatible
            with the knowledge base.
        """
        symbol_num = data_example.elements_num("pred_pseudo_label")
        max_revision_num = self._get_max_revision_num(self.max_revision, symbol_num)

        if self.use_zoopt:
            solution = self._zoopt_get_solution(symbol_num, data_example, max_revision_num)
            revision_idx = np.where(solution.get_x() != 0)[0]
            candidates, reasoning_results = self.kb.revise_at_idx(
                pseudo_label=data_example.pred_pseudo_label,
                y=data_example.Y,
                x=data_example.X,
                revision_idx=revision_idx,
            )
        else:
            candidates, reasoning_results = self.kb.abduce_candidates(
                pseudo_label=data_example.pred_pseudo_label,
                y=data_example.Y,
                x=data_example.X,
                max_revision_num=max_revision_num,
                require_more_revision=self.require_more_revision,
            )

        candidate = self._get_one_candidate(data_example, candidates, reasoning_results)
        return candidate

    def batch_abduce(self, data_examples: ListData) -> List[List[Any]]:
        """
        Perform abductive reasoning on the given prediction data examples.
        For detailed information, refer to ``abduce``.
        """
        abduced_pseudo_label = [self.abduce(data_example) for data_example in data_examples]
        data_examples.abduced_pseudo_label = abduced_pseudo_label
        return abduced_pseudo_label

    def batch_supervised_abduce(self, data_examples: ListData) -> List[List[Any]]:
        """
        Perform abductive reasoning on the given prediction data examples, using supervised data
        when gt_pseudo_label is given.
        """
        abduced_pseudo_label = [
            (
                data_example.gt_pseudo_label
                if data_example.gt_pseudo_label
                else self.abduce(data_example)
            )
            for data_example in data_examples
        ]
        data_examples.abduced_pseudo_label = abduced_pseudo_label
        return abduced_pseudo_label

    def __call__(self, data_examples: ListData) -> List[List[Any]]:
        return self.batch_abduce(data_examples)


# =============================================================================
# A3BL: Ambiguity-Aware Abductive Learning
#
# Reference: https://github.com/Hao-Yuan-He/A3BL
# =============================================================================


class A3BLReasoner(Reasoner):
    """
    Reasoner for minimizing the inconsistency between the knowledge base and learning models.

    Parameters
    ----------
    kb : class KBBase
        The knowledge base to be used for reasoning.
    dist_func : Union[str, Callable], optional
        The distance function used to determine the cost list between each
        candidate and the given prediction. The cost is also referred to as a consistency
        measure, wherein the candidate with the lowest cost is selected as the final
        abduced label. It can be either a string representing a predefined distance
        function or a callable function. The available predefined distance functions:
        'hamming' | 'confidence' | 'avg_confidence' | 'similarity' | 'rejection'.
        See :class:`Reasoner` for the full description of each option.
        Defaults to 'confidence'.
    idx_to_label : dict, optional
        A mapping from index in the base model to label. If not provided, a default
        order-based index to label mapping is created. Defaults to None.
    max_revision : Union[int, float], optional
        The upper limit on the number of revisions for each data example when
        performing abductive reasoning. If float, denotes the fraction of the total
        length that can be revised. A value of -1 implies no restriction on the
        number of revisions. Defaults to -1.
    require_more_revision : int, optional
        Specifies additional number of revisions permitted beyond the minimum required
        when performing abductive reasoning. Defaults to 0.
    use_zoopt : bool, optional
        Whether to use ZOOpt library during abductive reasoning. Defaults to False.
    topK : int, optional
        Number of top-ranked candidates to keep when forming the soft label. ``-1``
        keeps all candidates. Defaults to 16.
    temperature : float, optional
        Softmax temperature used when aggregating candidate probabilities into a
        soft label. Lower values produce sharper distributions. Defaults to 0.2.
    multi_label : bool, optional
        Whether the underlying task is multi-label (each symbol is a binary vector
        rather than a single class index). Defaults to False.
    """

    def __init__(
        self,
        kb,
        dist_func="confidence",
        idx_to_label=None,
        max_revision: Union[int, float] = -1,
        require_more_revision: int = 0,
        use_zoopt: bool = False,
        topK: int = 16,
        temperature: float = 0.2,
        multi_label: bool = False,
    ):
        super().__init__(
            kb, dist_func, idx_to_label, max_revision, require_more_revision, use_zoopt
        )
        import torch

        self.topK = topK
        self.temperature = temperature
        self.class_num = len(self.kb.pseudo_label_list)
        self.multi_label = multi_label
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _confidence_dist(
        self, pred_probs: np.ndarray, candidate_idxs: List[List[Any]], temp: float = 1.0
    ) -> np.ndarray:
        from scipy.special import softmax

        candidates_array = np.array(candidate_idxs)
        _, symbol_num = candidates_array.shape
        row_indices = np.arange(symbol_num)[:, np.newaxis]
        selected_probs = pred_probs[row_indices, candidates_array.T]
        candidate_probs = np.sum(selected_probs, axis=0) / temp
        return softmax(candidate_probs)

    def _confidence_dist_multi_label(
        self, pred_probs: np.ndarray, candidate_idxs: List[List[Any]], temp: float = 1.0
    ) -> np.ndarray:
        from scipy.special import softmax

        candidate_probs = pred_probs @ np.array(candidate_idxs).T / temp
        return softmax(candidate_probs.squeeze(axis=0))

    def _candidates_idxs(self, candidates: List[List[Any]]):
        return [[self.label_to_idx[x] for x in c] for c in candidates]

    def _topk(
        self, candidates: List[Any], candidate_probs: np.ndarray, K: int = -1
    ) -> Tuple[List[List[Any]], List[Any]]:
        """
        Performs a top-k selection from the candidate_set based on candidate_probs.
        If `K` is set to -1, all candidates are chosen.
        Returns a tuple containing the selected candidates and their corresponding probabilities.
        """
        import heapq

        if K == -1 or len(candidates) <= K:
            return candidates, candidate_probs

        # Iterate over all candidates and maintain a heap of size K with the largest probabilities
        heap = []
        for i, (candidate, prob) in enumerate(zip(candidates, candidate_probs)):
            if i < K:
                heapq.heappush(heap, (prob, candidate))
            else:
                if prob > heap[0][0]:
                    heapq.heappop(heap)
                    heapq.heappush(heap, (prob, candidate))

        # Extract top-k elements from the heap,
        # and reverse them to get the highest probabilities first
        topk_probs, topk_candidates = zip(*heap)
        return list(topk_candidates), list(topk_probs)

    def multi_label_aggregate(self, candidates: List[List[int]], candidate_probs: List[float]):
        """
        An multi-label version of A3BL.
        """
        import torch

        with torch.no_grad():
            symbol_num = len(candidates[0])
            aggregate_label = torch.zeros(size=(symbol_num, 1))
            for candidate, prob in zip(candidates, candidate_probs):
                for i, item in enumerate(candidate):
                    if item == 1:
                        aggregate_label[i] += prob
        return list(aggregate_label.unbind(1))

    def aggregate(self, candidates: List[List[int]], candidate_probs: List[float]):
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            candidates_tensor = torch.tensor(candidates, device=self.device, dtype=torch.long)
            probs_tensor = torch.tensor(candidate_probs, device=self.device, dtype=torch.float32)
            one_hot = F.one_hot(candidates_tensor, num_classes=self.class_num).float()  # [N, M, C]
            weighted_one_hot = one_hot * probs_tensor.unsqueeze(-1).unsqueeze(-1)  # [N, M, C]
            aggregate_label = weighted_one_hot.sum(dim=0)  # [M, C]
        return [tensor.cpu() for tensor in aggregate_label.unbind(0)]

    def abduce(self, data_example: ListData) -> Tuple[List[Any], List[Any]]:
        """
        Perform abduction and get a soft label distribution aggregated from
        all valid candidates that satisfy the underlying rules.

        Parameters
        ----------
        data_example : ListData
            Data example.

        Returns
        -------
        soft_label : List[Any]
            Soft label aggregated from the top-k valid candidates.
        pseudo_label : List[Any]
            Hard pseudo-label revision (the top-1 candidate) that is
            consistent with the knowledge base.
        """
        max_revision_num = data_example.elements_num("pred_pseudo_label")
        max_revision_num = self._get_max_revision_num(self.max_revision, max_revision_num)
        candidates, _ = self.kb.abduce_candidates(
            pseudo_label=data_example.pred_pseudo_label,
            y=data_example.Y,
            x=data_example.X,
            max_revision_num=max_revision_num,
            require_more_revision=self.require_more_revision,
        )

        if len(candidates) == 0:
            return [], []

        confidence_dist_cal = (
            self._confidence_dist if not self.multi_label else self._confidence_dist_multi_label
        )

        candidate_probs = confidence_dist_cal(
            data_example.pred_prob, self._candidates_idxs(candidates), self.temperature
        )
        topk_candidates, topk_candidates_probs = self._topk(candidates, candidate_probs, self.topK)
        aggregated_labels = (
            self.aggregate(topk_candidates, topk_candidates_probs)
            if not self.multi_label
            else self.multi_label_aggregate(topk_candidates, topk_candidates_probs)
        )
        return aggregated_labels, topk_candidates[0]

    def batch_abduce(self, data_examples: ListData) -> List[List[Any]]:
        """
        Perform abductive reasoning on the given prediction data examples.
        For detailed information, refer to ``abduce``.
        """
        abduced_soft_label, abduced_pseudo_label = zip(
            *[self.abduce(data_example) for data_example in data_examples]
        )
        data_examples.abduced_soft_label = abduced_soft_label
        data_examples.abduced_pseudo_label = abduced_pseudo_label
        return abduced_soft_label

    def __call__(self, data_examples: ListData) -> List[List[Any]]:
        return self.batch_abduce(data_examples)


# =============================================================================
# Verification Learning
#
# Walks the per-symbol probability lattice in descending joint-probability
# order, collecting the first top_k assignments that satisfy the knowledge
# base. Reference: https://github.com/VerificationLearning/VerificationLearning
# =============================================================================


def enumerate_label_assignments(
    pred_prob: np.ndarray, max_iter: int = 10000
) -> Iterator[Tuple[List[int], float, List[float]]]:
    """
    Yield label-index assignments for a single data example in descending
    joint-probability order. The walk is a Lawler-style best-first search:
    each state is the tuple of per-symbol rank indices, and successors are
    generated by advancing any one symbol to its next-best class.

    Parameters
    ----------
    pred_prob : np.ndarray
        Per-symbol probability matrix with shape ``(num_symbols, num_classes)``.
    max_iter : int, optional
        Hard cap on the number of yields. Defaults to 10000.

    Yields
    ------
    labels : List[int]
        Class indices for each symbol.
    joint_prob : float
        Product of the chosen per-symbol probabilities.
    per_symbol_probs : List[float]
        The chosen probability for each symbol.
    """
    pred_prob = np.asarray(pred_prob, dtype=float)
    num_symbols, num_classes = pred_prob.shape
    if num_symbols == 0:
        return

    sorted_indices = np.argsort(-pred_prob, axis=1)
    sorted_probs = np.take_along_axis(pred_prob, sorted_indices, axis=1)

    initial_state = (0,) * num_symbols
    initial_prob = float(np.prod(sorted_probs[:, 0]))

    seen = {initial_state}
    heap: List[Tuple[float, Tuple[int, ...]]] = [(-initial_prob, initial_state)]
    yields = 0

    while heap and yields < max_iter:
        neg_prob, state = heapq.heappop(heap)
        joint_prob = -neg_prob
        labels = [int(sorted_indices[i, state[i]]) for i in range(num_symbols)]
        per_symbol_probs = [float(sorted_probs[i, state[i]]) for i in range(num_symbols)]
        yield labels, joint_prob, per_symbol_probs
        yields += 1

        for sym in range(num_symbols):
            next_rank = state[sym] + 1
            if next_rank >= num_classes:
                continue
            new_state = state[:sym] + (next_rank,) + state[sym + 1:]
            if new_state in seen:
                continue
            seen.add(new_state)
            current_p = sorted_probs[sym, state[sym]]
            next_p = sorted_probs[sym, next_rank]
            if current_p <= 0:
                new_joint = 0.0
            else:
                new_joint = joint_prob * (next_p / current_p)
            heapq.heappush(heap, (-new_joint, new_state))


def top_k_satisfying(
    pred_prob: np.ndarray,
    predicate: Callable[[List[Any]], bool],
    top_k: int = 1,
    max_iter: int = 10000,
    idx_to_label: Optional[dict] = None,
) -> Tuple[List[List[Any]], List[float]]:
    """
    Walk label assignments in descending joint-probability order and return
    the first ``top_k`` that satisfy ``predicate``. If none is found within
    ``max_iter`` iterations the single highest-probability assignment is
    returned as a fallback so callers always receive a usable label.

    Parameters
    ----------
    pred_prob : np.ndarray
        Per-symbol probability matrix with shape ``(num_symbols, num_classes)``.
    predicate : Callable[[List[Any]], bool]
        Function called on each candidate label sequence; truthy means the
        candidate is consistent with the knowledge base.
    top_k : int, optional
        Maximum number of satisfying candidates to return. Defaults to 1.
    max_iter : int, optional
        Hard cap on enumeration steps. Defaults to 10000.
    idx_to_label : dict, optional
        Optional mapping from class index to pseudo-label. When omitted, the
        raw class indices are returned.

    Returns
    -------
    candidates : List[List[Any]]
        Label assignments that satisfy ``predicate`` (or the fallback).
    probs : List[float]
        Joint probability of each returned candidate.
    """
    matches: List[List[Any]] = []
    probs: List[float] = []
    fallback: Optional[Tuple[List[Any], float]] = None

    for labels_idx, joint_prob, _ in enumerate_label_assignments(pred_prob, max_iter):
        labels = (
            [idx_to_label[i] for i in labels_idx] if idx_to_label is not None else labels_idx
        )
        if fallback is None:
            fallback = (labels, joint_prob)
        if predicate(labels):
            matches.append(labels)
            probs.append(joint_prob)
            if len(matches) >= top_k:
                break

    if not matches and fallback is not None:
        matches.append(fallback[0])
        probs.append(fallback[1])
    return matches, probs


class VerificationReasoner:
    """
    Reasoner used by :class:`~ablkit.bridge.VerificationBridge`. Rather than
    picking a single best candidate via a distance function, it enumerates
    the top ``top_k`` label assignments that satisfy the knowledge base,
    ordered by joint probability. The bridge then trains the model on each
    of those candidates.

    Parameters
    ----------
    kb : KBBase
        The knowledge base used to verify candidates. ``kb.logic_forward``
        must return the reasoning result so it can be compared with each
        data example's ``Y``.
    top_k : int, optional
        Number of satisfying candidates to enumerate per example.
        Defaults to 1.
    max_iter : int, optional
        Maximum number of enumeration steps per example before giving up
        and returning the fallback. Defaults to 10000.
    idx_to_label : dict, optional
        A mapping from base-model index to pseudo-label. If omitted a
        default order-based mapping is built from ``kb.pseudo_label_list``.
    """

    def __init__(
        self,
        kb: KBBase,
        top_k: int = 1,
        max_iter: int = 10000,
        idx_to_label: Optional[dict] = None,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1.")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1.")
        self.kb = kb
        self.top_k = top_k
        self.max_iter = max_iter
        if idx_to_label is None:
            idx_to_label = dict(enumerate(kb.pseudo_label_list))
        self.idx_to_label = idx_to_label
        self.label_to_idx = {label: idx for idx, label in idx_to_label.items()}

    def top_k_candidates(
        self, pred_prob: np.ndarray, y: Any
    ) -> Tuple[List[List[Any]], List[float]]:
        """
        Return up to ``top_k`` label assignments for one data example whose
        ``kb.logic_forward`` matches ``y``.
        """

        def predicate(labels: List[Any]) -> bool:
            return self.kb.logic_forward(labels) == y

        return top_k_satisfying(
            pred_prob,
            predicate,
            top_k=self.top_k,
            max_iter=self.max_iter,
            idx_to_label=self.idx_to_label,
        )

    def batch_top_k(self, data_examples) -> List[List[List[Any]]]:
        """
        Run :meth:`top_k_candidates` on every example in ``data_examples``.
        Stores the result on ``data_examples.top_k_candidates`` and
        ``data_examples.top_k_probs``. Returns the list of per-example
        candidate lists.
        """
        all_candidates: List[List[List[Any]]] = []
        all_probs: List[List[float]] = []
        for data_example in data_examples:
            cands, probs = self.top_k_candidates(data_example.pred_prob, data_example.Y)
            all_candidates.append(cands)
            all_probs.append(probs)
        data_examples.top_k_candidates = all_candidates
        data_examples.top_k_probs = all_probs
        return all_candidates
