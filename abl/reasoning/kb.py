from abc import ABC, abstractmethod
import bisect
import numpy as np

from collections import defaultdict
from itertools import product, combinations

from abl.utils.utils import flatten, reform_idx, hamming_dist, check_equal, to_hashable, hashable_to_list

from multiprocessing import Pool

from functools import lru_cache
import pyswip

class KBBase(ABC):
    """
    Base class for reasoner.

    Attributes
    ----------
    pseudo_label_list : list
        List of possible pseudo labels.
    max_err : float, optional
        The upper tolerance limit when comparing the similarity between a candidate result  
        and the ground truth. Especially relevant for regression problems where exact matches 
        might not be feasible. Default to 0.
    use_cache : bool, optional
        Whether to use a cache for previously abduced candidates to speed up subsequent 
        operations. Defaults to True.
        
    Notes
    -----
    Users creating there own KB should inherit from this base class. For the inherited 
    subclass, it's mandatory to provide `pseudo_label_list` and override the `logic_forward`
    function. After that, other operations (e.g. how to perform abductive reasoning) 
    will be automatically set up.
    """
    def __init__(self, pseudo_label_list, max_err=0, use_cache=True):
        if not isinstance(pseudo_label_list, list):
            raise TypeError("pseudo_label_list should be list")
        self.pseudo_label_list = pseudo_label_list
        self.max_err = max_err
        self.use_cache = use_cache            

    @abstractmethod
    def logic_forward(self, pseudo_labels):
        pass

    def abduce_candidates(self, pred_pseudo_label, y, max_revision_num, require_more_revision=0):
        """
        Perform abductive reasoning to get a candidate consistent with the knowledge base.

        Parameters
        ----------
        pred_pseudo_label : List[Any]
            Predicted pseudo label.
        y : any
            Ground truth.
        max_revision_num : int
            The upper limit on the number of revisions.
        require_more_revision : int, optional
            Specifies additional number of revisions permitted beyond the minimum required.  
            Defaults to 0.

        Returns
        -------
        List[List[Any]]
            A list of candidates, i.e. revised pseudo label that are consistent with the 
            knowledge base.
        """
        if not self.use_cache:
            return self._abduce_by_search(pred_pseudo_label, y, 
                                          max_revision_num, require_more_revision)
        else:    
            return self._abduce_by_search_cache(to_hashable(pred_pseudo_label), 
                                                to_hashable(y), 
                                                max_revision_num, require_more_revision)
    
    def revise_at_idx(self, pred_pseudo_label, y, revision_idx):
        """
        Revise the predicted pseudo label at specified index positions.

        Parameters
        ----------
        pred_pseudo_label : List[Any]
            Predicted pseudo label.
        y : Any
            Ground truth.
        revision_idx : array-like
            Indices of where revisions should be made to the predicted pseudo label.
        """
        candidates = []
        abduce_c = product(self.pseudo_label_list, repeat=len(revision_idx))
        for c in abduce_c:
            candidate = pred_pseudo_label.copy()
            for i, idx in enumerate(revision_idx):
                candidate[idx] = c[i]
            if check_equal(self.logic_forward(candidate), y, self.max_err):
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
        y : any
            Ground truth.
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
            if revision_num == 0 and check_equal(self.logic_forward(pred_pseudo_label), 
                                                 y, 
                                                 self.max_err):
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
        
class ground_KB(KBBase):
    def __init__(self, pseudo_label_list, GKB_len_list=None, max_err=0):
        super().__init__(pseudo_label_list, max_err)
        
        self.GKB_len_list = GKB_len_list
        self.base = {}
        X, Y = self._get_GKB()
        for x, y in zip(X, Y):
            self.base.setdefault(len(x), defaultdict(list))[y].append(x)
            
    # For parallel version of _get_GKB
    def _get_XY_list(self, args):
        pre_x, post_x_it = args[0], args[1]
        XY_list = []
        for post_x in post_x_it:
            x = (pre_x,) + post_x
            y = self.logic_forward(x)
            if y is not None:
                XY_list.append((x, y))
        return XY_list

    # Parallel _get_GKB
    def _get_GKB(self):
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
        return self._abduce_by_GKB(pred_pseudo_label, y, max_revision_num, require_more_revision)
    
    def _find_candidate_GKB(self, pred_pseudo_label, y):
        if self.max_err == 0:
            return self.base[len(pred_pseudo_label)][y]
        else:
            potential_candidates = self.base[len(pred_pseudo_label)]
            key_list = list(potential_candidates.keys())
            key_idx = bisect.bisect_left(key_list, y)
            
            all_candidates = []
            for idx in range(key_idx - 1, 0, -1):
                k = key_list[idx]
                if abs(k - y) <= self.max_err:
                    all_candidates.extend(potential_candidates[k])
                else:
                    break
                
            for idx in range(key_idx, len(key_list)):
                k = key_list[idx]
                if abs(k - y) <= self.max_err:
                    all_candidates.extend(potential_candidates[k])
                else:
                    break
            return all_candidates
    
    def _abduce_by_GKB(self, pred_pseudo_label, y, max_revision_num, require_more_revision):
        if self.base == {} or len(pred_pseudo_label) not in self.GKB_len_list:
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

    def _dict_len(self, dic):
        if not self.GKB_flag:
            return 0
        else:
            return sum(len(c) for c in dic.values())

    def __len__(self):
        if not self.GKB_flag:
            return 0
        else:
            return sum(self._dict_len(v) for v in self.base.values())
        
        

class prolog_KB(KBBase):
    def __init__(self, pseudo_label_list, pl_file, max_err=0):
        super().__init__(pseudo_label_list, max_err)
        self.prolog = pyswip.Prolog()
        self.prolog.consult(pl_file)

    def logic_forward(self, pseudo_labels):
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
        
        # TODO：不知道有没有更简洁的方法
        regex = r"'P\d+'"
        return re.sub(regex, lambda x: x.group().replace("'", ""), str(revision_pred_pseudo_label))
    
    def get_query_string(self, pred_pseudo_label, y, revision_idx):
        query_string = "logic_forward("
        query_string += self._revision_pred_pseudo_label(pred_pseudo_label, revision_idx)
        key_is_none_flag = y is None or (type(y) == list and y[0] is None)
        query_string += ",%s)." % y if not key_is_none_flag else ")."
        return query_string
    
    def revise_at_idx(self, pred_pseudo_label, y, revision_idx):
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
