import numpy as np
from zoopt import Dimension, Objective, Parameter, Opt
from ..utils.utils import (
    confidence_dist,
    flatten,
    reform_list,
    hamming_dist,
)


class ReasonerBase:
    """
    Base class for reasoner.

    Parameters
    ----------
    kb : class KBBase
        The knowledge base to be used for reasoning.
    dist_func : str, optional
        The distance function to be used when determining the cost list between each
        candidate and the given prediction. Valid options include: "confidence" (default) |
        "hamming". For "confidence", it calculates the distance between the prediction
        and candidate based on confidence derived from the predicted probability in the
        data sample.For "hamming", it directly calculates the Hamming distance between
        the predicted pseudo label in the data sample and candidate.
    mapping : dict, optional
        A mapping from index in the base model to label. If not provided, a default
        order-based mapping is created.
    max_revision : int or float, optional
        The upper limit on the number of revisions for each data sample when
        performing abductive reasoning. If float, denotes the fraction of the total
        length that can be revised. A value of -1 implies no restriction on the
        number of revisions. Defaults to -1.
    require_more_revision : int, optional
        Specifies additional number of revisions permitted beyond the minimum required
        when performing abductive reasoning. Defaults to 0.
    use_zoopt : bool, optional
        Whether to use the Zoopt library during abductive reasoning. Defaults to False.
    """

    def __init__(
        self,
        kb,
        dist_func="confidence",
        mapping=None,
        max_revision=-1,
        require_more_revision=0,
        use_zoopt=False,
    ):
        if dist_func not in ["hamming", "confidence"]:
            raise NotImplementedError(
                'Valid options for dist_func include "hamming" and "confidence"'
            )

        self.kb = kb
        self.dist_func = dist_func
        self.use_zoopt = use_zoopt
        self.max_revision = max_revision
        self.require_more_revision = require_more_revision

        if mapping is None:
            self.mapping = {index: label for index, label in enumerate(self.kb.pseudo_label_list)}
        else:
            if not isinstance(mapping, dict):
                raise TypeError("mapping should be dict")
            for key, value in mapping.items():
                if not isinstance(key, int):
                    raise ValueError("All keys in the mapping must be integers")
                if value not in self.kb.pseudo_label_list:
                    raise ValueError("All values in the mapping must be in the pseudo_label_list")
            self.mapping = mapping
        self.remapping = dict(zip(self.mapping.values(), self.mapping.keys()))

    def _get_one_candidate(self, data_sample, candidates):
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
            cost_array = self._get_cost_list(data_sample, candidates)
            candidate = candidates[np.argmin(cost_array)]
            return candidate

    def _get_cost_list(self, data_sample, candidates):
        """
        Get the list of costs between each candidate and the given data sample. The list is
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
        """
        if self.dist_func == "hamming":
            return hamming_dist(data_sample.pred_pseudo_label, candidates)

        elif self.dist_func == "confidence":
            candidates = [[self.remapping[x] for x in c] for c in candidates]
            return confidence_dist(data_sample.pred_prob, candidates)

    def zoopt_get_solution(self, symbol_num, data_sample, max_revision_num):
        """
        Get the optimal solution using the Zoopt library. The solution is a list of
        boolean values, where '1' (True) indicates the indices chosen to be revised.

        Parameters
        ----------
        symbol_num : int
            Number of total symbols.
        data_sample : ListData
            Data sample.
        max_revision_num : int
            Specifies the maximum number of revisions allowed.
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

    def zoopt_revision_score(self, symbol_num, data_sample, sol):
        """
        Get the revision score for a solution. A lower score suggests that the Zoopt library
        has a higher preference for this solution.
        """
        revision_idx = np.where(sol.get_x() != 0)[0]
        candidates = self.kb.revise_at_idx(
            data_sample.pred_pseudo_label, data_sample.Y, revision_idx
        )
        if len(candidates) > 0:
            return np.min(self._get_cost_list(data_sample, candidates))
        else:
            return symbol_num

    def _constrain_revision_num(self, solution, max_revision_num):
        """
        Constrain that the total number of revisions chosen by the solution does not exceed
        maximum number of revisions allowed.
        """
        x = solution.get_x()
        return max_revision_num - x.sum()

    def _get_max_revision_num(self, max_revision, symbol_num):
        """
        Get the maximum revision number according to input `max_revision`.
        """
        if not isinstance(max_revision, (int, float)):
            raise TypeError("Parameter must be of type int or float.")

        if max_revision == -1:
            return symbol_num
        elif isinstance(max_revision, float):
            if not (0 <= max_revision <= 1):
                raise ValueError("If max_revision is a float, it must be between 0 and 1.")
            return round(symbol_num * max_revision)
        else:
            if max_revision < 0:
                raise ValueError("If max_revision is an int, it must be non-negative.")
            return max_revision

    def abduce(self, data_sample):
        """
        Perform abductive reasoning on the given data sample.

        Parameters
        ----------
        data_sample : ListData
            Data sample.

        Returns
        -------
        List[Any]
            A revised pseudo label through abductive reasoning, which is compatible with the
            knowledge base.
        """
        symbol_num = data_sample.elements_num("pred_pseudo_label")
        max_revision_num = self._get_max_revision_num(self.max_revision, symbol_num)

        if self.use_zoopt:
            solution = self.zoopt_get_solution(symbol_num, data_sample, max_revision_num)
            revision_idx = np.where(solution != 0)[0]
            candidates = self.kb.revise_at_idx(
                data_sample.pred_pseudo_label, data_sample.Y, revision_idx
            )
        else:
            candidates = self.kb.abduce_candidates(
                data_sample.pred_pseudo_label,
                data_sample.Y,
                max_revision_num,
                self.require_more_revision,
            )

        candidate = self._get_one_candidate(data_sample, candidates)
        return candidate

    def batch_abduce(self, data_samples):
        """
        Perform abductive reasoning on the given prediction data samples.
        For detailed information, refer to `abduce`.
        """
        abduced_pseudo_label = [self.abduce(data_sample) for data_sample in data_samples]
        data_samples.abduced_pseudo_label = abduced_pseudo_label
        return abduced_pseudo_label

    def __call__(self, data_samples):
        return self.batch_abduce(data_samples)
