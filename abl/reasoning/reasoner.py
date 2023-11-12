from typing import Any, List, Mapping, Optional

import numpy as np

from ..structures import ListData
from ..utils import (Cache, calculate_revision_num, confidence_dist,
                     hamming_dist)
from .base_kb import BaseKB
from .search_engine import BFS, BaseSearchEngine


class ReasonerBase:
    def __init__(
        self,
        kb: BaseKB,
        dist_func: str = "confidence",
        mapping: Optional[Mapping] = None,
        search_engine: Optional[BaseSearchEngine] = None,
        use_cache: bool = False,
        cache_file: Optional[str] = None,
        cache_size: Optional[int] = 4096,
    ):
        """
        Base class for all reasoner in the ABL system.

        Parameters
        ----------
        kb : BaseKB
            The knowledge base to be used for reasoning.
        dist_func : str, optional
            The distance function to be used. Can be "hamming" or "confidence". Default is "confidence".
        mapping : dict, optional
            A mapping of indices to labels. If None, a default mapping is generated.
        use_zoopt : bool, optional
            Whether to use the Zoopt library for optimization. Default is False.

        Raises
        ------
        NotImplementedError
            If the specified distance function is neither "hamming" nor "confidence".
        """

        if not isinstance(kb, BaseKB):
            raise ValueError("The kb should be of type BaseKB.")
        self.kb = kb

        if dist_func not in ["hamming", "confidence"]:
            raise NotImplementedError(f"The distance function '{dist_func}' is not implemented.")
        self.dist_func = dist_func

        if mapping is None:
            self.mapping = {index: label for index, label in enumerate(self.kb.pseudo_label_list)}
        else:
            if not isinstance(mapping, dict):
                raise ValueError("mapping must be of type dict")

            for key, value in mapping.items():
                if not isinstance(key, int):
                    raise ValueError("All keys in the mapping must be integers")

                if value not in self.kb.pseudo_label_list:
                    raise ValueError("All values in the mapping must be in the pseudo_label_list")

            self.mapping = mapping
        self.remapping = dict(zip(self.mapping.values(), self.mapping.keys()))

        if search_engine is None:
            self.search_engine = BFS()
        else:
            if not isinstance(search_engine, BaseSearchEngine):
                raise ValueError("The search_engine should be of type BaseSearchEngine.")
            else:
                self.search_engine = search_engine

        self.use_cache = use_cache
        self.cache_file = cache_file
        if self.use_cache:
            if not hasattr(self, "get_key"):
                raise NotImplementedError("If use_cache is True, get_key should be implemented.")
            key_func = self.get_key
        else:
            key_func = lambda x: x
        self.cache = Cache[ListData, List[List[Any]]](
            func=self.abduce,
            cache=self.use_cache,
            cache_file=self.cache_file,
            key_func=key_func,
            max_size=cache_size,
        )

    def _get_dist_list(self, data_sample: ListData, candidates: List[List[Any]]):
        """
        Get the list of costs between each pseudo label and candidate.

        Parameters
        ----------
        pred_pseudo_label : list
            The pseudo label to be used for computing costs of candidates.
        pred_prob : list
            Probabilities of the predictions. Used when distance function is "confidence".
        candidates : list
            List of candidate abduction result.

        Returns
        -------
        numpy.ndarray
            Array of computed costs for each candidate.
        """
        if self.dist_func == "hamming":
            return hamming_dist(data_sample["pred_pseudo_label"][0], candidates)

        elif self.dist_func == "confidence":
            candidates = [[self.remapping[x] for x in c] for c in candidates]
            return confidence_dist(data_sample["pred_prob"][0], candidates)

    def select(self, data_sample: ListData, candidates: List[List[Any]]):
        """
        Get one candidate. If multiple candidates exist, return the one with minimum cost.

        Parameters
        ----------
        pred_pseudo_label : list
            The pseudo label to be used for selecting a candidate.
        pred_prob : list
            Probabilities of the predictions.
        candidates : list
            List of candidate abduction result.

        Returns
        -------
        list
            The chosen candidate based on minimum cost.
            If no candidates, an empty list is returned.
        """
        if len(candidates) == 0:
            return []
        elif len(candidates) == 1:
            return candidates[0]
        else:
            cost_array = self._get_dist_list(data_sample, candidates)
            candidate = candidates[np.argmin(cost_array)]
            return candidate

    def abduce(
        self,
        data_sample: ListData,
        max_revision: int = -1,
        require_more_revision: int = 0,
    ):
        """
        Perform revision by abduction on the given data.

        Parameters
        ----------
        pred_prob : list
            List of probabilities for predicted results.
        pred_pseudo_label : list
            List of predicted pseudo labels.
        y : any
            Ground truth for the predicted results.
        max_revision : int or float, optional
            Maximum number of revisions to use. If float, represents the fraction of total revisions to use.
            If -1, any revisions are allowed. Defaults to -1.
        require_more_revision : int, optional
            Number of additional revisions to require. Defaults to 0.

        Returns
        -------
        list
            The abduced revisions.
        """
        symbol_num = data_sample.elements_num("pred_pseudo_label")
        max_revision_num = calculate_revision_num(max_revision, symbol_num)
        data_sample.set_metainfo(dict(symbol_num=symbol_num))

        if hasattr(self.kb, "abduce_candidates"):
            candidates = self.kb.abduce_candidates(
                data_sample, max_revision_num, require_more_revision
            )
        elif hasattr(self.kb, "revise_at_idx"):
            candidates = []
            gen = self.search_engine.generator(
                data_sample,
                max_revision_num=max_revision_num,
                require_more_revision=require_more_revision,
            )
            send_signal = True
            for revision_idx in gen:
                candidates.extend(self.kb.revise_at_idx(data_sample, revision_idx))
                if len(candidates) > 0 and send_signal:
                    try:
                        revision_idx = gen.send("success")
                        candidates.extend(self.kb.revise_at_idx(data_sample, revision_idx))
                        send_signal = False
                    except StopIteration:
                        break
        else:
            raise NotImplementedError(
                "The kb should either implement abduce_candidates or revise_at_idx."
            )

        candidate = self.select(data_sample, candidates)
        return candidate

    def batch_abduce(
        self,
        data_samples: ListData,
        max_revision: int = -1,
        require_more_revision: int = 0,
    ):
        """
        Perform abduction on the given data in batches.

        Parameters
        ----------
        pred_prob : list
            List of probabilities for predicted results.
        pred_pseudo_label : list
            List of predicted pseudo labels.
        Y : list
            List of ground truths for the predicted results.
        max_revision : int or float, optional
            Maximum number of revisions to use. If float, represents the fraction of total revisions to use.
            If -1, use all revisions. Defaults to -1.
        require_more_revision : int, optional
            Number of additional revisions to require. Defaults to 0.

        Returns
        -------
        list
            The abduced revisions in batches.
        """
        abduced_pseudo_label = [
            self.cache.get(
                data_sample,
                max_revision=max_revision,
                require_more_revision=require_more_revision,
            )
            for data_sample in data_samples
        ]
        data_samples.abduced_pseudo_label = abduced_pseudo_label
        return abduced_pseudo_label
