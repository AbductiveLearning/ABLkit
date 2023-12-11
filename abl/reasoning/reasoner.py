import numpy as np
from zoopt import Dimension, Objective, Opt, Parameter
from typing import Callable, Any, List, Optional

from kb import KBBase
from ..structures import ListData
from ..utils.utils import confidence_dist, hamming_dist


class Reasoner:
    """
    Base class for reasoner.

    Parameters
    ----------
    kb : class KBBase
        The knowledge base to be used for reasoning.
    dist_func : str or Callable, optional
        The distance function to be used when determining the cost list between each
        candidate and the given prediction. Defaults to "confidence".
    mapping : Optional[dict], optional
        A mapping from index in the base model to label. If not provided, a default
        order-based mapping is created. Defaults to None.
    max_revision : int or float, optional
        The upper limit on the number of revisions for each data sample when
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
        dist_func: str or Callable = "confidence",
        mapping: Optional[dict] = None,
        max_revision: int or float = -1,
        require_more_revision: int = 0,
        use_zoopt: bool = False,
    ):
        self.kb = kb
        self.dist_func = dist_func
        self.use_zoopt = use_zoopt
        self.max_revision = max_revision
        self.require_more_revision = require_more_revision

        if mapping is None:
            self.mapping = {index: label for index, label in enumerate(self.kb.pseudo_label_list)}
        else:
            self._check_valid_mapping(mapping)
            self.mapping = mapping
        self.remapping = dict(zip(self.mapping.values(), self.mapping.keys()))

    def _check_valid_mapping(self, mapping):
        if not isinstance(mapping, dict):
            raise TypeError(f"mapping should be dict, got {type(mapping)}")
        for key, value in mapping.items():
            if not isinstance(key, int):
                raise ValueError(f"All keys in the mapping must be integers, got {key}")
            if value not in self.kb.pseudo_label_list:
                raise ValueError(f"All values in the mapping must be in the pseudo_label_list, got {value}")
    
    def _get_one_candidate(
        self, 
        data_sample: ListData, 
        candidates: List[List[Any]],
    ) -> List[Any]:
        """
        Due to the nondeterminism of abductive reasoning, there could be multiple candidates
        satisfying the knowledge base. When this happens, return one candidate that has the
        minimum cost. If no candidates are provided, an empty list is returned.

        Parameters
        ----------
        data_sample : ListData
            Data sample.
        candidates : List[List[Any]]
            Multiple compatible candidates.

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
            cost_array = self.get_cost_list(data_sample, candidates)
            candidate = candidates[np.argmin(cost_array)]
            return candidate

    def get_cost_list(
        self, 
        data_sample: ListData, 
        candidates: List[List[Any]],
    ) -> np.ndarray:
        """
        Get the list of costs between each candidate and the given data sample. 
        
        The list is
        calculated based on one of the following distance functions:
        - "hamming": Directly calculates the Hamming distance between the predicted pseudo
                     label in the data sample and candidate.
        - "confidence": Calculates the distance between the prediction and candidate based
                        on confidence derived from the predicted probability in the data
                        sample.

        Parameters
        ----------
        data_sample : ListData
            Data sample.
        candidates : List[List[Any]]
            Multiple compatible candidates.
        
        Returns
        -------
        np.ndarray
            A Numpy array representing list of costs.
        """
        if self.dist_func == "hamming":
            return hamming_dist(data_sample.pred_pseudo_label, candidates)

        elif self.dist_func == "confidence":
            candidates = [[self.remapping[x] for x in c] for c in candidates]
            return confidence_dist(data_sample.pred_prob, candidates)
        
        elif callable(self.dist_func):
            return self.dist_func(data_sample, candidates)

        else:
            raise ValueError("dist_func must be either a string or a callable function")


    def _zoopt_get_solution(
        self, 
        symbol_num: int, 
        data_sample: ListData, 
        max_revision_num: int,
    ) -> List[bool]:
        """
        Get the optimal solution using ZOOpt library. The solution is a list of
        boolean values, where '1' (True) indicates the indices chosen to be revised.

        Parameters
        ----------
        symbol_num : int
            Number of total symbols.
        data_sample : ListData
            Data sample.
        max_revision_num : int
            Specifies the maximum number of revisions allowed.
            
        Returns
        -------
        List[bool]
            The solution for ZOOpt library.
        """
        dimension = Dimension(size=symbol_num, regs=[[0, 1]] * symbol_num, tys=[False] * symbol_num)
        objective = Objective(
            lambda sol: self.zoopt_revision_score(symbol_num, data_sample, sol),
            dim=dimension,
            constraint=lambda sol: self._constrain_revision_num(sol, max_revision_num),
        )
        parameter = Parameter(budget=100, intermediate_result=False, autoset=True)
        solution = Opt.min(objective, parameter).get_x()
        return solution

    def zoopt_revision_score(
        self, 
        symbol_num: int, 
        data_sample: ListData, 
        sol: List[bool],
    ) -> int:
        """
        Get the revision score for a solution. A lower score suggests that ZOOpt library
        has a higher preference for this solution.
        
        Parameters
        ----------
        symbol_num : int
            Number of total symbols.
        data_sample : ListData
            Data sample.
        sol: List[bool]
            The solution for ZOOpt library.
            
        Returns
        -------
        int
            The revision score for the solution.
        """
        revision_idx = np.where(sol.get_x() != 0)[0]
        candidates = self.kb.revise_at_idx(
            data_sample.pred_pseudo_label, data_sample.Y, data_sample.X, revision_idx
        )
        if len(candidates) > 0:
            return np.min(self.get_cost_list(data_sample, candidates))
        else:
            return symbol_num

    def _constrain_revision_num(self, solution: List[bool], max_revision_num: int) -> int:
        """
        Constrain that the total number of revisions chosen by the solution does not exceed
        maximum number of revisions allowed.
        """
        x = solution.get_x()
        return max_revision_num - x.sum()

    def _get_max_revision_num(self, max_revision: int or float, symbol_num: int) -> int:
        """
        Get the maximum revision number according to input `max_revision`.
        """
        if not isinstance(max_revision, (int, float)):
            raise TypeError(f"Parameter must be of type int or float, got {type(max_revision)}")

        if max_revision == -1:
            return symbol_num
        elif isinstance(max_revision, float):
            if not (0 <= max_revision <= 1):
                raise ValueError(f"If max_revision is a float, it must be between 0 and 1, but got {max_revision}")
            return round(symbol_num * max_revision)
        else:
            if max_revision < 0:
                raise ValueError(f"If max_revision is an int, it must be non-negative, but got {max_revision}")
            return max_revision

    def abduce(self, data_sample: ListData) -> List[Any]:
        """
        Perform abductive reasoning on the given data sample.

        Parameters
        ----------
        data_sample : ListData
            Data sample.

        Returns
        -------
        List[Any]
            A revised pseudo label sample through abductive reasoning, which is compatible
            with the knowledge base.
        """
        symbol_num = data_sample.elements_num("pred_pseudo_label")
        max_revision_num = self._get_max_revision_num(self.max_revision, symbol_num)

        if self.use_zoopt:
            solution = self._zoopt_get_solution(symbol_num, data_sample, max_revision_num)
            revision_idx = np.where(solution != 0)[0]
            candidates = self.kb.revise_at_idx(
                data_sample.pred_pseudo_label, data_sample.Y, data_sample.X, revision_idx
            )
        else:
            candidates = self.kb.abduce_candidates(
                pseudo_label = data_sample.pred_pseudo_label,
                y = data_sample.Y,
                x = data_sample.X,
                max_revision_num = max_revision_num,
                require_more_revision = self.require_more_revision,
            )

        candidate = self._get_one_candidate(data_sample, candidates)
        return candidate

    def batch_abduce(self, data_samples: ListData) -> List[List[Any]]:
        """
        Perform abductive reasoning on the given prediction data samples.
        For detailed information, refer to `abduce`.
        """
        abduced_pseudo_label = [self.abduce(data_sample) for data_sample in data_samples]
        data_samples.abduced_pseudo_label = abduced_pseudo_label
        return abduced_pseudo_label

    def __call__(self, data_samples: ListData) -> List[List[Any]]:
        return self.batch_abduce(data_samples)
