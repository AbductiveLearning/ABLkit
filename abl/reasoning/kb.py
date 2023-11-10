from abc import ABC, abstractmethod
import bisect
from collections import defaultdict
from itertools import product, combinations
from multiprocessing import Pool
from functools import lru_cache

import numpy as np
import pyswip

from ..utils.utils import flatten, reform_idx, hamming_dist, to_hashable, hashable_to_list


class KBBase(ABC):
    """
    Base class for knowledge base.

    Parameters
    ----------
    pseudo_label_list : list
        List of possible pseudo labels.
    max_err : float, optional
        The upper tolerance limit when comparing the similarity between a candidate's logical 
        result and the ground truth. Especially relevant for regression problems where exact 
        matches might not be feasible. Defaults to 1e-10.
    use_cache : bool, optional
        Whether to use a cache for previously abduced candidates to speed up subsequent 
        operations. Defaults to True.
        
    Notes
    -----
    Users should inherit from this base class to build their own knowledge base. For the 
    user-build KB (an inherited subclass), it's only required for the user to provide the 
    `pseudo_label_list` and override the `logic_forward` function (specifying how to 
    perform logical reasoning). After that, other operations (e.g. how to perform abductive 
    reasoning) will be automatically set up. 
    """
    def __init__(self, pseudo_label_list, max_err=1e-10, use_cache=True):
        if not isinstance(pseudo_label_list, list):
            raise TypeError("pseudo_label_list should be list")
        self.pseudo_label_list = pseudo_label_list
        self.max_err = max_err
        self.use_cache = use_cache            

    @abstractmethod
    def logic_forward(self, pseudo_labels):
        """
        How to perform (deductive) logical reasoning, i.e. matching each pseudo label to 
        their logical result. Users are required to provide this.
        """
        pass

    def abduce_candidates(self, pred_pseudo_label, y, max_revision_num, require_more_revision=0):
        """
        Perform abductive reasoning to get a candidate consistent with the knowledge base.

        Parameters
        ----------
        pred_pseudo_label : List[Any]
            Predicted pseudo label.
        y : any
            Ground truth for the logical result.
        max_revision_num : int
            The upper limit on the number of revisions.
        require_more_revision : int, optional
            Specifies additional number of revisions permitted beyond the minimum required.  
            Defaults to 0.

        Returns
        -------
        List[List[Any]]
            A list of candidates, i.e. revised pseudo labels that are consistent with the 
            knowledge base.
        """
        if not self.use_cache:
            return self._abduce_by_search(pred_pseudo_label, y, 
                                          max_revision_num, require_more_revision)
        else:    
            return self._abduce_by_search_cache(to_hashable(pred_pseudo_label), 
                                                to_hashable(y), 
                                                max_revision_num, require_more_revision)
    
    def _check_equal(self, logic_result, y):
        """
        Check whether the logical result of a candidate is equal to the ground truth
        (or, within the maximum error allowed).
        """
        if logic_result == None:
            return False
        
        if isinstance(logic_result, (int, float)) and isinstance(y, (int, float)):
            return abs(logic_result - y) <= self.max_err
        else:
            return logic_result == y
         
    def revise_at_idx(self, pred_pseudo_label, y, revision_idx):
        """
        Revise the predicted pseudo label at specified index positions.

        Parameters
        ----------
        pred_pseudo_label : List[Any]
            Predicted pseudo label.
        y : Any
            Ground truth for the logical result.
        revision_idx : array-like
            Indices of where revisions should be made to the predicted pseudo label.
        """
        candidates = []
        abduce_c = product(self.pseudo_label_list, repeat=len(revision_idx))
        for c in abduce_c:
            candidate = pred_pseudo_label.copy()
            for i, idx in enumerate(revision_idx):
                candidate[idx] = c[i]
            if self._check_equal(self.logic_forward(candidate), y):
                candidates.append(candidate)
        return candidates

    def _revision(self, revision_num, pred_pseudo_label, y):
        """
        For a specified number of pseudo label to revise, iterate through all possible 
        indices to find any candidates that are consistent with the knowledge base.
        """
        new_candidates = []
        revision_idx_list = combinations(range(len(pred_pseudo_label)), revision_num)

        for revision_idx in revision_idx_list:
            candidates = self.revise_at_idx(pred_pseudo_label, y, revision_idx)
            new_candidates.extend(candidates)
        return new_candidates

    def _abduce_by_search(self, pred_pseudo_label, y, max_revision_num, require_more_revision):   
        """
        Perform abductive reasoning by exhastive search. Specifically, begin with 0 and   
        continuously increase the number of pseudo labels to revise, until candidates 
        that are consistent with the knowledge base are found.
        
        Parameters
        ----------
        pred_pseudo_label : List[Any]
            Predicted pseudo label.
        y : Any
            Ground truth for the logical result.
        max_revision_num : int
            The upper limit on the number of revisions.
        require_more_revision : int
            If larger than 0, then after having found any candidates consistent with the 
            knowledge base, continue to increase the number pseudo labels to revise to 
            get more possible consistent candidates.

        Returns
        -------
        List[List[Any]]
            A list of candidates, i.e. revised pseudo label that are consistent with the 
            knowledge base.
        """   
        candidates = []
        for revision_num in range(len(pred_pseudo_label) + 1):
            if revision_num == 0 and self._check_equal(self.logic_forward(pred_pseudo_label), y):
                candidates.append(pred_pseudo_label)
            elif revision_num > 0:
                candidates.extend(self._revision(revision_num, pred_pseudo_label, y))
            if len(candidates) > 0:
                min_revision_num = revision_num
                break
            if revision_num >= max_revision_num:
                return []

        for revision_num in range(min_revision_num + 1, min_revision_num + require_more_revision + 1):
            if revision_num > max_revision_num:
                return candidates
            candidates.extend(self._revision(revision_num, pred_pseudo_label, y))
        return candidates
    
    @lru_cache(maxsize=None)
    def _abduce_by_search_cache(self, pred_pseudo_label, y, max_revision_num, require_more_revision):
        """
        `_abduce_by_search` with cache.
        """
        pred_pseudo_label = hashable_to_list(pred_pseudo_label)
        y = hashable_to_list(y)
        return self._abduce_by_search(pred_pseudo_label, y, max_revision_num, require_more_revision)

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
    class initialization, stroing all potential candidates along with their respective 
    logical result. Ground KB can accelerate abductive reasoning in `abduce_candidates`. 

    Parameters
    ----------
    pseudo_label_list : list
        Refer to class `KBBase`.
    GKB_len_list : list
        List of possible lengths of pseudo label.
    max_err : float, optional
        Refer to class `KBBase`.
    
    Notes
    -----
    Users can also inherit from this class to build their own knowledge base. Similar 
    to `KBBase`, users are only required to provide the `pseudo_label_list` and override 
    the `logic_forward` function. Additionally, users should provide the `GKB_len_list`.
    After that, other operations (e.g. auto-construction of GKB, and how to perform 
    abductive reasoning) will be automatically set up.
    """
    def __init__(self, pseudo_label_list, GKB_len_list, max_err=0):
        super().__init__(pseudo_label_list, max_err)
        if not isinstance(GKB_len_list, list):
            raise TypeError("GKB_len_list should be list")
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
        Prebuild the GKB according to `pseudo_label_list` and `GKB_len_list`.
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
    
    def abduce_candidates(self, pred_pseudo_label, y, max_revision_num, require_more_revision=0):
        """
        Perform abductive reasoning by directly retrieving consistent candidates from 
        the prebuilt GKB. In this way, the time-consuming exhaustive search can be  
        avoided.
        This is an overridden function. For more information about the parameters and 
        returns, refer to the function of the same name in class `KBBase`.
        """
        if self.GKB == {} or len(pred_pseudo_label) not in self.GKB_len_list:
            return []
        
        all_candidates = self._find_candidate_GKB(pred_pseudo_label, y)
        if len(all_candidates) == 0:
            return []

        cost_list = hamming_dist(pred_pseudo_label, all_candidates)
        min_revision_num = np.min(cost_list)
        revision_num = min(max_revision_num, min_revision_num + require_more_revision)
        idxs = np.where(cost_list <= revision_num)[0]
        candidates = [all_candidates[idx] for idx in idxs]
        return candidates
    
    def _find_candidate_GKB(self, pred_pseudo_label, y):
        """
        Retrieve consistent candidates from the prebuilt GKB. If `max_err` is greater 
        than 0, return all candidates whose logical results fall within the 
        [y - max_err, y + max_err] range.
        """
        if self.max_err == 0:
            return self.GKB[len(pred_pseudo_label)][y]
        else:
            potential_candidates = self.GKB[len(pred_pseudo_label)]
            key_list = list(potential_candidates.keys())
            
            low_key = bisect.bisect_left(key_list, y - self.max_err)
            high_key = bisect.bisect_right(key_list, y + self.max_err)

            all_candidates = [candidate
                            for key in key_list[low_key:high_key]
                            for candidate in potential_candidates[key]]
            return all_candidates
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__} is a KB with "
            f"pseudo_label_list={self.pseudo_label_list!r}, "
            f"max_err={self.max_err!r}, "
            f"use_cache={self.use_cache!r}, "
            f"and has a prebuilt GKB with "
            f"GKB_len_list={self.GKB_len_list!r}."
        )


class PrologKB(KBBase):
    """
    Knowledge base given by a prolog (pl) file.

    Parameters
    ----------
    pseudo_label_list : list
        Refer to class `KBBase`.
    pl_file : 
        Prolog file containing the KB.
    max_err : float, optional
        Refer to class `KBBase`.
    
    Notes
    -----
    Users can also inherit from this class to build their own knowledge base. When using 
    this class, users are only required to provide the `pl_file`.
    """
    def __init__(self, pseudo_label_list, pl_file):
        super().__init__(pseudo_label_list)
        self.pl_file = pl_file
        self.prolog = pyswip.Prolog()
        self.prolog.consult(self.pl_file)

    def logic_forward(self, pseudo_labels):
        """
        Consult prolog with the query `logic_forward(pseudo_labels, Res).`, and set the 
        returned `Res` as the logical results. To use this default function, there must be 
        a Prolog `log_forward` method in the pl file to perform logical. reasoning. Otherwise, 
        users would override this function.
        """
        result = list(self.prolog.query("logic_forward(%s, Res)." % pseudo_labels))[0]['Res']
        if result == 'true':
            return True
        elif result == 'false':
            return False
        return result
    
    def _revision_pred_pseudo_label(self, pred_pseudo_label, revision_idx):
        import re
        revision_pred_pseudo_label = pred_pseudo_label.copy()
        revision_pred_pseudo_label = flatten(revision_pred_pseudo_label)
        
        for idx in revision_idx:
            revision_pred_pseudo_label[idx] = 'P' + str(idx)
        revision_pred_pseudo_label = reform_idx(revision_pred_pseudo_label, pred_pseudo_label)
        
        regex = r"'P\d+'"
        return re.sub(regex, lambda x: x.group().replace("'", ""), str(revision_pred_pseudo_label))
    
    def get_query_string(self, pred_pseudo_label, y, revision_idx):
        """
        Consult prolog with `logic_forward([kept_labels, Revise_labels], Res).`, and set 
        the returned `Revise_labels` together with the kept labels as the candidates. This is 
        a default fuction for demo, users would override this function to adapt to their own
        Prolog file. 
        """
        query_string = "logic_forward("
        query_string += self._revision_pred_pseudo_label(pred_pseudo_label, revision_idx)
        key_is_none_flag = y is None or (type(y) == list and y[0] is None)
        query_string += ",%s)." % y if not key_is_none_flag else ")."
        return query_string
    
    def revise_at_idx(self, pred_pseudo_label, y, revision_idx):
        """
        Revise the predicted pseudo label at specified index positions by querying Prolog.
        This is an overridden function. For more information about the parameters, refer to 
        the function of the same name in class `KBBase`.
        """
        candidates = []
        query_string = self.get_query_string(pred_pseudo_label, y, revision_idx)
        save_pred_pseudo_label = pred_pseudo_label
        pred_pseudo_label = flatten(pred_pseudo_label)
        abduce_c = [list(z.values()) for z in self.prolog.query(query_string)]
        for c in abduce_c:
            candidate = pred_pseudo_label.copy()
            for i, idx in enumerate(revision_idx):
                candidate[idx] = c[i]
            candidate = reform_idx(candidate, save_pred_pseudo_label)
            candidates.append(candidate)
        return candidates

    def __repr__(self):
        return (
            f"{self.__class__.__name__} is a KB with "
            f"pseudo_label_list={self.pseudo_label_list!r}, "
            f"defined by "
            f"Prolog file {self.pl_file!r}."
        )