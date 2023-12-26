import bisect
import inspect
import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import combinations, product
from multiprocessing import Pool
from typing import Any, Callable, List, Optional

import numpy as np

from ..utils.cache import abl_cache
from ..utils.logger import print_log
from ..utils.utils import flatten, hamming_dist, reform_list, to_hashable


class KBBase(ABC):
    """
    Base class for knowledge base.

    Parameters
    ----------
    pseudo_label_list : list
        List of possible pseudo-labels. It's recommended to arrange the pseudo-labels in this
        list so that each aligns with its corresponding index in the base model: the first with
        the 0th index, the second with the 1st, and so forth.
    max_err : float, optional
        The upper tolerance limit when comparing the similarity between the reasoning result of 
        pseudo-labels and the ground truth. This is only applicable when the reasoning
        result is of a numerical type. This is particularly relevant for regression problems where
        exact matches might not be feasible. Defaults to 1e-10.
    use_cache : bool, optional
        Whether to use abl_cache for previously abduced candidates to speed up subsequent
        operations. Defaults to True.
    key_func : Callable, optional
        A function employed for hashing in abl_cache. This is only operational when use_cache
        is set to True. Defaults to to_hashable.
    cache_size: int, optional
        The cache size in abl_cache. This is only operational when use_cache is set to
        True. Defaults to 4096.

    Notes
    -----
    Users should derive from this base class to build their own knowledge base. For the
    user-build KB (a derived subclass), it's only required for the user to provide the
    ``pseudo_label_list`` and override the ``logic_forward`` function (specifying how to
    perform logical reasoning). After that, other operations (e.g. how to perform abductive
    reasoning) will be automatically set up.
    """

    def __init__(
        self,
        pseudo_label_list: list,
        max_err: float = 1e-10,
        use_cache: bool = True,
        key_func: Callable = to_hashable,
        cache_size: int = 4096,
    ):
        if not isinstance(pseudo_label_list, list):
            raise TypeError(f"pseudo_label_list should be list, got {type(pseudo_label_list)}")
        self.pseudo_label_list = pseudo_label_list
        self.max_err = max_err

        self.use_cache = use_cache
        self.key_func = key_func
        self.cache_size = cache_size
        
        argspec = inspect.getfullargspec(self.logic_forward)
        self._num_args = len(argspec.args) - 1
        if self._num_args==2 and self.use_cache: # If the logic_forward function has 2 arguments, then disable cache
            self.use_cache = False
            print_log(
                "The logic_forward function has 2 arguments, so the cache is disabled. ",
                logger="current",
                level=logging.WARNING,
            )
            # TODO 添加半监督
            # TODO 添加consistency measure+max_err容忍错误

    @abstractmethod
    def logic_forward(self, pseudo_label: List[Any], x: Optional[List[Any]] = None) -> Any:
        """
        How to perform (deductive) logical reasoning, i.e. matching pseudo-labels to
        their reasoning result. Users are required to provide this.

        Parameters
        ----------
        pseudo_label : List[Any]
            Pseudo-labels of an example.
        x : Optional[List[Any]]
            The example. If deductive logical reasoning does not require any 
            information from the example, the overridden function provided by the user can omit 
            this parameter.
        
        Returns
        -------
        Any
            The reasoning result.
        """

    def abduce_candidates(
        self, 
        pseudo_label: List[Any], 
        y: Any, 
        x: List[Any], 
        max_revision_num: int, 
        require_more_revision: int,
    ) -> List[List[Any]]:
        """
        Perform abductive reasoning to get a candidate compatible with the knowledge base.

        Parameters
        ----------
        pseudo_label : List[Any]
            Pseudo-labels of an example (to be revised by abductive reasoning).
        y : Any
            Ground truth of the reasoning result for the example.
        x : List[Any]
            The example. If the information from the example
            is not required in the reasoning process, then this parameter will not have 
            any effect.
        max_revision_num : int
            The upper limit on the number of revised labels for each example.
        require_more_revision : int
            Specifies additional number of revisions permitted beyond the minimum required.

        Returns
        -------
        Tuple[List[List[Any]], List[Any]]
            A tuple of two element. The first element is a list of candidate revisions, i.e. revised
            pseudo-labels of the example. that are compatible with the knowledge base. The second element is 
            a list of reasoning results corresponding to each candidate, i.e., the outcome of the 
            logic_forward function.
        """
        return self._abduce_by_search(pseudo_label, y, x, max_revision_num, require_more_revision)

    def _check_equal(self, reasoning_result: Any, y: Any) -> bool:
        """
        Check whether the reasoning result of a pseduo label example is equal to the ground truth
        (or, within the maximum error allowed for numerical results).

        Returns
        -------
        bool
            The result of the check.
        """
        if reasoning_result is None:
            return False

        if isinstance(reasoning_result, (int, float)) and isinstance(y, (int, float)):
            return abs(reasoning_result - y) <= self.max_err
        else:
            return reasoning_result == y

    def revise_at_idx(
        self, 
        pseudo_label: List[Any], 
        y: Any, 
        x: List[Any], 
        revision_idx: List[int],
    ) -> List[List[Any]]:
        """
        Revise the pseudo-labels at specified index positions.

        Parameters
        ----------
        pseudo_label : List[Any]
            Pseudo-labels of an example (to be revised).
        y : Any
            Ground truth of the reasoning result for the example.
        x : List[Any]
            The example. If the information from the example
            is not required in the reasoning process, then this parameter will not have 
            any effect.
        revision_idx : List[int]
            A list specifying indices of where revisions should be made to the pseudo-labels.

        Returns
        -------
        Tuple[List[List[Any]], List[Any]]
            A tuple of two element. The first element is a list of candidate revisions, i.e. revised
            pseudo-labels of the example that are compatible with the knowledge base. The second element is 
            a list of reasoning results corresponding to each candidate, i.e., the outcome of the 
            logic_forward function.
        """
        candidates, reasoning_results = [], []
        abduce_c = product(self.pseudo_label_list, repeat=len(revision_idx))
        for c in abduce_c:
            candidate = pseudo_label.copy()
            for i, idx in enumerate(revision_idx):
                candidate[idx] = c[i]
            reasoning_result = self.logic_forward(candidate, *(x,) if self._num_args == 2 else ())
            if self._check_equal(reasoning_result, y):
                candidates.append(candidate); reasoning_results.append(reasoning_result)
        return candidates, reasoning_results

    def _revision(
        self, 
        revision_num: int, 
        pseudo_label: List[Any], 
        y: Any, 
        x: List[Any],
    ) -> List[List[Any]]:
        """
        For a specified number of labels in a pseudo-labels to revise, iterate through
        all possible indices to find any candidates that are compatible with the knowledge base.
        """
        new_candidates, new_reasoning_results = [], []
        revision_idx_list = combinations(range(len(pseudo_label)), revision_num)
        for revision_idx in revision_idx_list:
            candidates, reasoning_results = self.revise_at_idx(pseudo_label, y, x, revision_idx)
            new_candidates.extend(candidates); new_reasoning_results.extend(reasoning_results)
        return new_candidates, new_reasoning_results

    @abl_cache()
    def _abduce_by_search(
        self, 
        pseudo_label: List[Any], 
        y: Any, 
        x: List[Any], 
        max_revision_num: int, 
        require_more_revision: int,
    ) -> List[List[Any]]:
        """
        Perform abductive reasoning by exhastive search. Specifically, begin with 0 and
        continuously increase the number of labels to revise, until
        candidates that are compatible with the knowledge base are found.

        Parameters
        ----------
        pseudo_label : List[Any]
            Pseudo-labels of an example (to be revised).
        y : Any
            Ground truth of the reasoning result for the example.
        x : List[Any]
            The example. If the information from the example
            is not required in the reasoning process, then this parameter will not have 
            any effect.
        max_revision_num : int
            The upper limit on the number of revisions.
        require_more_revision : int
            If larger than 0, then after having found any candidates compatible with the
            knowledge base, continue to increase the number of labels to
            revise to get more possible compatible candidates.

        Returns
        -------
        Tuple[List[List[Any]], List[Any]]
            A tuple of two element. The first element is a list of candidate revisions, i.e. revised
            pseudo-labels of the example that are compatible with the knowledge base. The second element is 
            a list of reasoning results corresponding to each candidate, i.e., the outcome of the 
            logic_forward function.
        """
        candidates, reasoning_results = [], []
        for revision_num in range(len(pseudo_label) + 1):
            new_candidates, new_reasoning_results = self._revision(revision_num, pseudo_label, y, x)
            candidates.extend(new_candidates); reasoning_results.extend(new_reasoning_results)
            if len(candidates) > 0:
                min_revision_num = revision_num
                break
            if revision_num >= max_revision_num:
                return [], []

        for revision_num in range(
            min_revision_num + 1, min_revision_num + require_more_revision + 1
        ):
            if revision_num > max_revision_num:
                return candidates, reasoning_results
            new_candidates, new_reasoning_results = self._revision(revision_num, pseudo_label, y, x)
            candidates.extend(new_candidates); reasoning_results.extend(new_reasoning_results)
        return candidates, reasoning_results

    def __repr__(self):
        return (
            f"{self.__class__.__name__} is a KB with "
            f"pseudo_label_list={self.pseudo_label_list!r}, "
            f"max_err={self.max_err!r}, "
            f"use_cache={self.use_cache!r}."
        )


class GroundKB(KBBase):
    """
    Knowledge base with a ground KB (GKB). Ground KB is a knowledge base prebuilt upon
    class initialization, storing all potential candidates along with their respective
    reasoning result. Ground KB can accelerate abductive reasoning in ``abduce_candidates``.

    Parameters
    ----------
    pseudo_label_list : list
        Refer to class ``KBBase``.
    GKB_len_list : list
        List of possible lengths for pseudo-labels of an example.
    max_err : float, optional
        Refer to class ``KBBase``.

    Notes
    -----
    Users can also inherit from this class to build their own knowledge base. Similar
    to ``KBBase``, users are only required to provide the ``pseudo_label_list`` and override
    the ``logic_forward`` function. Additionally, users should provide the ``GKB_len_list``.
    After that, other operations (e.g. auto-construction of GKB, and how to perform
    abductive reasoning) will be automatically set up.
    """

    def __init__(self, pseudo_label_list, GKB_len_list, max_err=1e-10):
        super().__init__(pseudo_label_list, max_err)
        if not isinstance(GKB_len_list, list):
            raise TypeError("GKB_len_list should be list, but got {type(GKB_len_list)}")
        if self._num_args==2:
            raise NotImplementedError(f"GroundKB only supports 1-argument logic_forward, but got {self._num_args}-argument logic_forward")
        self.GKB_len_list = GKB_len_list
        self.GKB = {}
        X, Y = self._get_GKB()
        for x, y in zip(X, Y):
            self.GKB.setdefault(len(x), defaultdict(list))[y].append(x)

    def _get_XY_list(self, args):
        pre_x, post_x_it = args[0], args[1]
        XY_list = []
        for post_x in post_x_it:
            x = (pre_x,) + post_x
            y = self.logic_forward(x)
            if y is not None:
                XY_list.append((x, y))
        return XY_list

    def _get_GKB(self):
        """
        Prebuild the GKB according to ``pseudo_label_list`` and ``GKB_len_list``.
        """
        X, Y = [], []
        for length in self.GKB_len_list:
            arg_list = []
            for pre_x in self.pseudo_label_list:
                post_x_it = product(self.pseudo_label_list, repeat=length - 1)
                arg_list.append((pre_x, post_x_it))
            with Pool(processes=len(arg_list)) as pool:
                ret_list = pool.map(self._get_XY_list, arg_list)
            for XY_list in ret_list:
                if len(XY_list) == 0:
                    continue
                part_X, part_Y = zip(*XY_list)
                X.extend(part_X)
                Y.extend(part_Y)
        if Y and isinstance(Y[0], (int, float)):
            X, Y = zip(*sorted(zip(X, Y), key=lambda pair: pair[1]))
        return X, Y

    def abduce_candidates(
        self, 
        pseudo_label: List[Any], 
        y: Any, 
        x: List[Any], 
        max_revision_num: int, 
        require_more_revision: int,
    ) -> List[List[Any]]:
        """
        Perform abductive reasoning by directly retrieving compatible candidates from
        the prebuilt GKB. In this way, the time-consuming exhaustive search can be
        avoided.

        Parameters
        ----------
        pseudo_label : List[Any]
            Pseudo-labels of an example (to be revised by abductive reasoning).
        y : Any
            Ground truth of the reasoning result for the example.
        x : List[Any]
            The example (unused in GroundKB).
        max_revision_num : int
            The upper limit on the number of revised labels for each example.
        require_more_revision : int
            Specifies additional number of revisions permitted beyond the minimum required.

        Returns
        -------
        Tuple[List[List[Any]], List[Any]]
            A tuple of two element. The first element is a list of candidate revisions, i.e. revised
            pseudo-labels of THE example that are compatible with the knowledge base. The second element is 
            a list of reasoning results corresponding to each candidate, i.e., the outcome of the 
            logic_forward function.
        """
        if self.GKB == {} or len(pseudo_label) not in self.GKB_len_list:
            return [], []

        all_candidates, all_reasoning_results = self._find_candidate_GKB(pseudo_label, y)
        if len(all_candidates) == 0:
            return [], []

        cost_list = hamming_dist(pseudo_label, all_candidates)
        min_revision_num = np.min(cost_list)
        revision_num = min(max_revision_num, min_revision_num + require_more_revision)
        idxs = np.where(cost_list <= revision_num)[0]
        candidates = [all_candidates[idx] for idx in idxs]
        reasoning_results = [all_reasoning_results[idx] for idx in idxs]
        return candidates, reasoning_results

    def _find_candidate_GKB(self, pseudo_label: List[Any], y: Any) -> List[List[Any]]:
        """
        Retrieve compatible candidates from the prebuilt GKB. For numerical reasoning results,
        return all candidates and their corresponding reasoning results which fall within the
        [y - max_err, y + max_err] range.
        """
        if isinstance(y, (int, float)):
            potential_candidates = self.GKB[len(pseudo_label)]
            key_list = list(potential_candidates.keys())

            low_key = bisect.bisect_left(key_list, y - self.max_err)
            high_key = bisect.bisect_right(key_list, y + self.max_err)

            all_candidates, all_reasoning_results = [], []
            for key in key_list[low_key:high_key]:
                for candidate in potential_candidates[key]:
                    all_candidates.append(candidate); all_reasoning_results.append(key)
        else:
            all_candidates = self.GKB[len(pseudo_label)][y]
            all_reasoning_results = [y] * len(all_candidates)
        return all_candidates, all_reasoning_results

    def __repr__(self):
        GKB_info_parts = []
        for i in self.GKB_len_list:
            num_candidates = len(self.GKB[i]) if i in self.GKB else 0
            GKB_info_parts.append(f"{num_candidates} candidates of length {i}")
        GKB_info = ", ".join(GKB_info_parts)

        return (
            f"{self.__class__.__name__} is a KB with "
            f"pseudo_label_list={self.pseudo_label_list!r}, "
            f"max_err={self.max_err!r}, "
            f"use_cache={self.use_cache!r}. "
            f"It has a prebuilt GKB with "
            f"GKB_len_list={self.GKB_len_list!r}, "
            f"and there are "
            f"{GKB_info}"
            f" in the GKB."
        )


class PrologKB(KBBase):
    """
    Knowledge base provided by a Prolog (.pl) file.

    Parameters
    ----------
    pseudo_label_list : list
        Refer to class ``KBBase``.
    pl_file :
        Prolog file containing the KB.
    max_err : float, optional
        Refer to class ``KBBase``.

    Notes
    -----
    Users can instantiate this class to build their own knowledge base. During the
    instantiation, users are only required to provide the ``pseudo_label_list`` and ``pl_file``.
    To use the default logic forward and abductive reasoning methods in this class, in the
    Prolog (.pl) file, there needs to be a rule which is strictly formatted as
    ``logic_forward(Pseudo_labels, Res).``, e.g., ``logic_forward([A,B], C) :- C is A+B``.
    For specifics, refer to the ``logic_forward`` and ``get_query_string`` functions in this
    class. Users are also welcome to override related functions for more flexible support.
    """

    def __init__(self, pseudo_label_list: List[Any], pl_file: str):
        import pyswip
        
        super().__init__(pseudo_label_list)
        
        try:
            import pyswip
        except (IndexError, ImportError):
            print("A Prolog-based knowledge base is in use. Please install Swi-Prolog \
                   using the command 'sudo apt-get install swi-prolog' for Linux users, \
                   or download it from https://www.swi-prolog.org/Download.html for Windows and Mac users.")
        
        self.prolog = pyswip.Prolog()
        self.pl_file = pl_file
        if not os.path.exists(self.pl_file):
            raise FileNotFoundError(f"The Prolog file {self.pl_file} does not exist.")
        self.prolog.consult(self.pl_file)

    def logic_forward(self, pseudo_label: List[Any]) -> Any:
        """
        Consult prolog with the query ``logic_forward(pseudo_labels, Res).``, and set the
        returned ``Res`` as the reasoning results. To use this default function, there must be
        a ``logic_forward`` method in the pl file to perform reasoning.
        Otherwise, users would override this function.

        Parameters
        ----------
        pseudo_label : List[Any]
            Pseudo-labels of an example.
        """
        result = list(self.prolog.query("logic_forward(%s, Res)." % pseudo_label))[0]["Res"]
        if result == "true":
            return True
        elif result == "false":
            return False
        return result

    def _revision_pseudo_label(
        self,
        pseudo_label: List[Any],
        revision_idx: List[int],
    ) -> List[Any]:
        import re

        revision_pseudo_label = pseudo_label.copy()
        revision_pseudo_label = flatten(revision_pseudo_label)

        for idx in revision_idx:
            revision_pseudo_label[idx] = "P" + str(idx)
        revision_pseudo_label = reform_list(revision_pseudo_label, pseudo_label)

        regex = r"'P\d+'"
        return re.sub(regex, lambda x: x.group().replace("'", ""), str(revision_pseudo_label))

    def get_query_string(
        self, 
        pseudo_label: List[Any], 
        y: Any, 
        x: List[Any],
        revision_idx: List[int],
    ) -> str:
        """
        Get the query to be used for consulting Prolog.
        This is a default function for demo, users would override this function to adapt to
        their own Prolog file. In this demo function, return query
        ``logic_forward([kept_labels, Revise_labels], Res).``.

        Parameters
        ----------
        pseudo_label : List[Any]
            Pseudo-labels of an example (to be revised by abductive reasoning).
        y : Any
            Ground truth of the reasoning result for the example.
        x : List[Any]
            The corresponding input example. If the information from the input 
            is not required in the reasoning process, then this parameter will not have 
            any effect.
        revision_idx : List[int]
            A list specifying indices of where revisions should be made to the pseudo-labels.

        Returns
        -------
        str
            A string of the query.
        """
        query_string = "logic_forward("
        query_string += self._revision_pseudo_label(pseudo_label, revision_idx)
        key_is_none_flag = y is None or (isinstance(y, list) and y[0] is None)
        query_string += ",%s)." % y if not key_is_none_flag else ")."
        return query_string

    def revise_at_idx(
        self, 
        pseudo_label: List[Any], 
        y: Any, 
        x: List[Any], 
        revision_idx: List[int],
    ) -> List[List[Any]]:
        """
        Revise the pseudo-labels at specified index positions by querying Prolog.

        Parameters
        ----------
        pseudo_label : List[Any]
            Pseudo-labels of an example (to be revised).
        y : Any
            Ground truth of the reasoning result for the example.
        x : List[Any]
            The corresponding input example. If the information from the input 
            is not required in the reasoning process, then this parameter will not have 
            any effect.
        revision_idx : List[int]
            A list specifying indices of where revisions should be made to the pseudo-labels.

        Returns
        -------
        Tuple[List[List[Any]], List[Any]]
            A list of candidates, i.e. revised pseudo-labels of the example that are compatible with the
            knowledge base.
            A tuple of two element. The first element is a list of candidate revisions, i.e. revised
            pseudo-labels of the example that are compatible with the knowledge base. The second element is 
            a list of reasoning results corresponding to each candidate, i.e., the outcome of the 
            logic_forward function.
        """
        candidates, reasoning_results = [], []
        query_string = self.get_query_string(pseudo_label, y, x, revision_idx)
        save_pseudo_label = pseudo_label
        pseudo_label = flatten(pseudo_label)
        abduce_c = [list(z.values()) for z in self.prolog.query(query_string)]
        for c in abduce_c:
            candidate = pseudo_label.copy()
            for i, idx in enumerate(revision_idx):
                candidate[idx] = c[i]
            candidate = reform_list(candidate, save_pseudo_label)
            candidates.append(candidate); reasoning_results.append(y)
        return candidates, reasoning_results

    def __repr__(self):
        return (
            f"{self.__class__.__name__} is a KB with "
            f"pseudo_label_list={self.pseudo_label_list!r}, "
            f"defined by "
            f"Prolog file {self.pl_file!r}."
        )
