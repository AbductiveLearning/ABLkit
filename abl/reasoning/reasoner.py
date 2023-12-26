import inspect
from typing import Any, Callable, List, Optional, Union

import numpy as np
from zoopt import Dimension, Objective, Opt, Parameter, Solution

from ..data.structures import ListData
from ..reasoning import KBBase
from ..utils.utils import confidence_dist, hamming_dist


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
        'hamming' | 'confidence'. 'hamming': directly calculates the Hamming
        distance between the predicted pseudo-label in the data example and each
        candidate, 'confidence': calculates the distance between the prediction
        and each candidate based on confidence derived from the predicted probability
        in the data example. The callable function should have the signature
        dist_func(data_example, candidates, candidate_idxs, reasoning_results) and must return a cost list. Each element
        in this cost list should be a numerical value representing the cost for each
        candidate, and the list should have the same length as candidates.
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
            if dist_func not in ["hamming", "confidence"]:
                raise NotImplementedError(
                    f'Valid options for predefined dist_func include "hamming" and "confidence", but got {dist_func}.'
                )
            return
        elif callable(dist_func):
            params = inspect.signature(dist_func).parameters.values()
            if len(params) != 4:
                raise ValueError(
                    f"User-defined dist_func must have exactly four parameters, but got {len(params)}."
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
                    f"All values in the idx_to_label must be in the pseudo_label_list, but got {value}."
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
            Multiple compatible candidates.
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
            Multiple compatible candidates.
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
            candidates = [[self.label_to_idx[x] for x in c] for c in candidates]
            return confidence_dist(data_example.pred_prob, candidates)
        else:
            candidate_idxs = [[self.label_to_idx[x] for x in c] for c in candidates]
            cost_list = self.dist_func(data_example, candidates, candidate_idxs, reasoning_results)
            if len(cost_list) != len(candidates):
                raise ValueError(
                    f"The length of the array returned by dist_func must be equal to the number of candidates. "
                    f"Expected length {len(candidates)}, but got {len(cost_list)}."
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
        parameter = Parameter(budget=self.zoopt_budget(symbol_num), intermediate_result=False, autoset=True)
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
        Set the budget for ZOOpt optimization. The function, in its default implementation, 
        returns a fixed budget value of 100. However, it can be adjusted to return other fixed 
        values, or a dynamic budget based on the number of symbols, if desired. For example, one might choose to 
        set the budget as 100 times symbol_num.

        Parameters
        ----------
        symbol_num : int
            The number of symbols to be considered in the ZOOpt optimization process. Although this parameter
            can be used to compute a dynamic optimization budget, by default it is not utilized in the
            calculation.

        Returns
        -------
        int
            The budget for ZOOpt optimization. By default, this is a fixed value of 100, 
            irrespective of the symbol_num value.
        """
        return 100
        

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
        elif isinstance(max_revision, float):
            if not (0 <= max_revision <= 1):
                raise ValueError(
                    f"If max_revision is a float, it must be between 0 and 1, but got {max_revision}"
                )
            return round(symbol_num * max_revision)
        else:
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

    def __call__(self, data_examples: ListData) -> List[List[Any]]:
        return self.batch_abduce(data_examples)
