import numpy as np
from zoopt import Dimension, Objective, Parameter, Opt
from abl.utils.utils import (
    confidence_dist,
    flatten,
    reform_idx,
    hamming_dist,
    calculate_revision_num,
)


class ReasonerBase:
    """
    Base class for reasoner.

    Attributes
    ----------
    kb :
        The knowledge base to be used for reasoning.
    dist_func : str, optional
        The distance function to be used when determining the cost list between each 
        candidate and the given prediction. Valid options include: `"hamming"` (default) 
        | `"confidence"`. Any other options will raise a `NotImplementedError`. For 
        detailed explanations of these options, refer to `_get_cost_list`.
    mapping : dict, optional
        A mapping from index to label. If not provided, a default order-based mapping is 
        created.
    use_zoopt : bool, optional
        Whether to use the Zoopt library during abductive reasoning. Default to False.
    """
        
    def __init__(self, kb, dist_func="hamming", mapping=None, use_zoopt=False):
        if not (dist_func == "hamming" or dist_func == "confidence"):
            raise NotImplementedError  

        self.kb = kb
        self.dist_func = dist_func
        self.use_zoopt = use_zoopt
        if mapping is None:
            self.mapping = {
                label: index for index, label in enumerate(self.kb.pseudo_label_list)
            }
        else:
            if not isinstance(mapping, dict):
                raise TypeError("mapping should be dict")
            self.mapping = mapping
        self.remapping = dict(zip(self.mapping.values(), self.mapping.keys()))

    def _get_one_candidate(self, pred_pseudo_label, pred_prob, candidates):
        """
        Due to the nondeterminism of abductive reasoning, there could be multiple candidates 
        satisfying the knowledge base. When this happens, return one candidate that has the 
        minimum cost. If no candidates are provided, an empty list is returned.
        
        Parameters
        ----------
        pred_pseudo_label : List[Any]
            Predicted pseudo label to be used for selecting a candidate.
        pred_prob : List[List[Any]]
            Predicted probabilities of the prediction (Each sublist contains the probability 
            distribution over all pseudo labels).
        candidates : List[List[Any]]
            Multiple candidate abduction results. 
        """
        if len(candidates) == 0:
            return []
        elif len(candidates) == 1:
            return candidates[0]
        else:
            cost_array = self._get_cost_list(pred_pseudo_label, pred_prob, candidates)
            candidate = candidates[np.argmin(cost_array)]
            return candidate
    
    def _get_cost_list(self, pred_pseudo_label, pred_prob, candidates):
        """
        Get the list of costs between each candidate and the given prediction. The list is 
        calculated based on one of the following distance functions:
        - "hamming": Directly calculates the Hamming distance between the predicted pseudo 
                     label and candidate.
        - "confidence": Calculates the distance between the prediction and candidate based 
                        on confidence derived from the predicted probability.
        
        Parameters
        ----------
        pred_pseudo_label : List[Any]
            Predicted pseudo label.
        pred_prob : List[List[Any]]
            Predicted probabilities of the prediction (Each sublist contains the probability 
            distribution over all pseudo labels). Used when distance function is "confidence".
        candidates : List[List[Any]]
            Multiple candidate abduction results.
        """
        if self.dist_func == "hamming":
            return hamming_dist(pred_pseudo_label, candidates)

        elif self.dist_func == "confidence":
            candidates = [[self.remapping[x] for x in c] for c in candidates]
            return confidence_dist(pred_prob, candidates)


    def zoopt_get_solution(
        self, symbol_num, pred_pseudo_label, pred_prob, y, max_revision_num
    ):
        """
        Get the optimal solution using the Zoopt library. The solution is a list of 
        boolean values, where '1' (True) indicates the indices chosen to be revised.

        Parameters
        ----------
        symbol_num : int
            Number of total symbols.
        pred_pseudo_label : List[Any]
            Predicted pseudo label.
        pred_prob : List[List[Any]]
            Predicted probabilities of the prediction (Each sublist contains the probability 
            distribution over all pseudo labels).
        y : Any
            Ground truth.
        max_revision_num : int
            Specifies the maximum number of revisions allowed.
        """
        dimension = Dimension(
            size=symbol_num, regs=[[0, 1]] * symbol_num, tys=[False] * symbol_num
        )
        objective = Objective(
            lambda sol: self.zoopt_revision_score(
                symbol_num, pred_pseudo_label, pred_prob, y, sol
            ),
            dim=dimension,
            constraint=lambda sol: self._constrain_revision_num(sol, max_revision_num),
        )
        parameter = Parameter(budget=100, intermediate_result=False, autoset=True)
        solution = Opt.min(objective, parameter).get_x()
        return solution
    
    def zoopt_revision_score(self, symbol_num, pred_pseudo_label, pred_prob, y, sol):
        """
        Get the revision score for a solution. A lower score suggests that the Zoopt library 
        has a higher preference for this solution.
        """
        revision_idx = np.where(sol.get_x() != 0)[0]
        candidates = self.revise_at_idx(pred_pseudo_label, y, revision_idx)
        if len(candidates) > 0:
            return np.min(self._get_cost_list(pred_pseudo_label, pred_prob, candidates))
        else:
            return symbol_num

    def _constrain_revision_num(self, solution, max_revision_num):
        """
        Constrain that the total number of revisions chosen by the solution does not exceed 
        maximum number of revisions allowed.
        """
        x = solution.get_x()
        return max_revision_num - x.sum()
    
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
        return self.kb.revise_at_idx(pred_pseudo_label, y, revision_idx)

    def abduce(
        self, pred_prob, pred_pseudo_label, y, max_revision=-1, require_more_revision=0
    ):
        """
        Perform abductive reasoning on the given prediction data.

        Parameters
        ----------
        pred_prob : List[List[Any]]
            Predicted probabilities of the prediction (Each sublist contains the probability 
            distribution over all pseudo labels).
        pred_pseudo_label : List[Any]
            Predicted pseudo label.
        y : any
            Ground truth.
        max_revision : int or float, optional
            The upper limit on the number of revisions. If float, denotes the fraction of the 
            total length that can be revised. A value of -1 implies no restriction on the number 
            of revisions. Default to -1.
        require_more_revision : int, optional
            Specifies additional number of revisions permitted beyond the minimum required.  
            Defaults to 0.

        Returns
        -------
        List[Any]
            A revised pseudo label through abductive reasoning, which is consistent with the
            knowledge base.
        """
        symbol_num = len(flatten(pred_pseudo_label))
        max_revision_num = calculate_revision_num(max_revision, symbol_num)

        if self.use_zoopt:
            solution = self.zoopt_get_solution(
                symbol_num, pred_pseudo_label, pred_prob, y, max_revision_num
            )
            revision_idx = np.where(solution != 0)[0]
            candidates = self.revise_at_idx(pred_pseudo_label, y, revision_idx)
        else:
            candidates = self.kb.abduce_candidates(
                pred_pseudo_label, y, max_revision_num, require_more_revision
            )

        candidate = self._get_one_candidate(pred_pseudo_label, pred_prob, candidates)
        return candidate

    def batch_abduce(
        self, pred_probs, pred_pseudo_labels, Ys, max_revision=-1, require_more_revision=0
    ):
        """
        Perform abductive reasoning on the given prediction data in batches.
        For detailed information, refer to `abduce`.
        """
        return [
            self.abduce(
                pred_prob, pred_pseudo_label, Y, max_revision, require_more_revision
            )
            for pred_prob, pred_pseudo_label, Y in zip(
                pred_probs, pred_pseudo_labels, Ys
            )
        ]

    # def _batch_abduce_helper(self, args):
    #     z, prob, y, max_revision, require_more_revision = args
    #     return self.abduce((z, prob, y), max_revision, require_more_revision)

    # def batch_abduce(self, Z, Y, max_revision=-1, require_more_revision=0):
    #     with Pool(processes=os.cpu_count()) as pool:
    #         results = pool.map(self._batch_abduce_helper, [(z, prob, y, max_revision, require_more_revision) for z, prob, y in zip(Z['cls'], Z['prob'], Y)])
    #     return results

    def __call__(
        self, pred_prob, pred_pseudo_label, Y, max_revision=-1, require_more_revision=0
    ):
        return self.batch_abduce(
            pred_prob, pred_pseudo_label, Y, max_revision, require_more_revision
        )




if __name__ == "__main__":
    from kb import KBBase, ground_KB, prolog_KB

    prob1 = [[[0, 0.99, 0.01, 0, 0, 0, 0, 0, 0, 0],
              [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]]
    
    prob2 = [[[0, 0, 0.01, 0, 0, 0, 0, 0.99, 0, 0],
              [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]]

    class add_KB(KBBase):
        def __init__(self, pseudo_label_list=list(range(10)),
                           use_cache=True):
            super().__init__(pseudo_label_list, use_cache=use_cache)

        def logic_forward(self, nums):
            return sum(nums)
        
    class add_ground_KB(ground_KB):
        def __init__(self, pseudo_label_list=list(range(10)), 
                           GKB_len_list=[2]):
            super().__init__(pseudo_label_list, GKB_len_list)

        def logic_forward(self, nums):
            return sum(nums)
        
    def test_add(reasoner):
        res = reasoner.batch_abduce(prob1, [[1, 1]], [8], max_revision=2, require_more_revision=0)
        print(res)
        res = reasoner.batch_abduce(prob2, [[1, 1]], [8], max_revision=2, require_more_revision=0)
        print(res)
        res = reasoner.batch_abduce(prob1, [[1, 1]], [17], max_revision=2, require_more_revision=0)
        print(res)
        res = reasoner.batch_abduce(prob1, [[1, 1]], [17], max_revision=1, require_more_revision=0)
        print(res)
        res = reasoner.batch_abduce(prob1, [[1, 1]], [20], max_revision=2, require_more_revision=0)
        print(res)
        print()

    print("add_KB with GKB:")
    kb = add_ground_KB()
    reasoner = ReasonerBase(kb, "confidence")
    test_add(reasoner)

    print("add_KB without GKB:")
    kb = add_KB()
    reasoner = ReasonerBase(kb, "confidence")
    test_add(reasoner)

    print("add_KB without GKB, no cache")
    kb = add_KB(use_cache=False)
    reasoner = ReasonerBase(kb, "confidence")
    test_add(reasoner)

    print("prolog_KB with add.pl:")
    kb = prolog_KB(pseudo_label_list=list(range(10)),
                   pl_file="examples/mnist_add/datasets/add.pl")
    reasoner = ReasonerBase(kb, "confidence")
    test_add(reasoner)

    print("prolog_KB with add.pl using zoopt:")
    kb = prolog_KB(
        pseudo_label_list=list(range(10)),
        pl_file="examples/mnist_add/datasets/add.pl",
    )
    reasoner = ReasonerBase(kb, "confidence", use_zoopt=True)
    test_add(reasoner)

    print("add_KB with multiple inputs at once:")
    multiple_prob = [[
                        [0, 0.99, 0.01, 0, 0, 0, 0, 0, 0, 0],
                        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                    ],
                    [
                        [0, 0, 0.01, 0, 0, 0, 0, 0.99, 0, 0],
                        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                    ]]

    kb = add_KB()
    reasoner = ReasonerBase(kb, "confidence")
    res = reasoner.batch_abduce(
        multiple_prob,
        [[1, 1], [1, 2]],
        [4, 8],
        max_revision=2,
        require_more_revision=0,
    )
    print(res)
    res = reasoner.batch_abduce(
        multiple_prob,
        [[1, 1], [1, 2]],
        [4, 8],
        max_revision=2,
        require_more_revision=1,
    )
    print(res)
    print()

    class HWF_KB(KBBase):
        def __init__(
            self,
            pseudo_label_list=["1", "2", "3", "4", "5", "6", "7", "8", "9",
                               "+", "-", "times", "div"],
            max_err=1e-3,
        ):
            super().__init__(pseudo_label_list, max_err)

        def _valid_candidate(self, formula):
            if len(formula) % 2 == 0:
                return False
            for i in range(len(formula)):
                if i % 2 == 0 and formula[i] not in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                    return False
                if i % 2 != 0 and formula[i] not in ["+", "-", "times", "div"]:
                    return False
            return True

        def logic_forward(self, formula):
            if not self._valid_candidate(formula):
                return np.inf
            mapping = {str(i): str(i) for i in range(1, 10)}
            mapping.update({"+": "+", "-": "-", "times": "*", "div": "/"})
            formula = [mapping[f] for f in formula]
            return eval("".join(formula))
    
    class HWF_ground_KB(ground_KB):
        def __init__(
            self,
            pseudo_label_list=["1", "2", "3", "4", "5", "6", "7", "8", "9",
                               "+", "-", "times", "div"],
            GKB_len_list=[1, 3, 5, 7],
            max_err=1e-3,
        ):
            super().__init__(pseudo_label_list, GKB_len_list, max_err)

        def _valid_candidate(self, formula):
            if len(formula) % 2 == 0:
                return False
            for i in range(len(formula)):
                if i % 2 == 0 and formula[i] not in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                    return False
                if i % 2 != 0 and formula[i] not in ["+", "-", "times", "div"]:
                    return False
            return True

        def logic_forward(self, formula):
            if not self._valid_candidate(formula):
                return np.inf
            mapping = {str(i): str(i) for i in range(1, 10)}
            mapping.update({"+": "+", "-": "-", "times": "*", "div": "/"})
            formula = [mapping[f] for f in formula]
            return eval("".join(formula))
    
    def test_hwf(reasoner):
        res = reasoner.batch_abduce(
            [None],
            [["5", "+", "2"]],
            [3],
            max_revision=2,
            require_more_revision=0,
        )
        print(res)
        res = reasoner.batch_abduce(
            [None],
            [["5", "+", "9"]],
            [65],
            max_revision=3,
            require_more_revision=0,
        )
        print(res)
        res = reasoner.batch_abduce(
            [None],
            [["5", "8", "8", "8", "8"]],
            [3.17],
            max_revision=5,
            require_more_revision=3,
        )
        print(res)
        print()
    
    def test_hwf_multiple(reasoner, max_revisions):
        res = reasoner.batch_abduce(
            [None, None],
            [["5", "+", "2"], ["5", "+", "9"]],
            [3, 64],
            max_revision=max_revisions[0],
            require_more_revision=0,
        )
        print(res)
        res = reasoner.batch_abduce(
            [None, None],
            [["5", "+", "2"], ["5", "+", "9"]],
            [3, 64],
            max_revision=max_revisions[1],
            require_more_revision=0,
        )
        print(res)
        res = reasoner.batch_abduce(
            [None, None],
            [["5", "+", "2"], ["5", "+", "9"]],
            [3, 65],
            max_revision=max_revisions[2],
            require_more_revision=0,
        )
        print(res)
        print()

    print("HWF_KB with GKB, max_err=0.1")
    kb = HWF_ground_KB(GKB_len_list=[1, 3, 5], max_err=0.1)
    reasoner = ReasonerBase(kb, "hamming")
    test_hwf(reasoner)

    print("HWF_KB without GKB, max_err=0.1")
    kb = HWF_KB(max_err=0.1)
    reasoner = ReasonerBase(kb, "hamming")
    test_hwf(reasoner)

    print("HWF_KB with GKB, max_err=1")
    kb = HWF_ground_KB(GKB_len_list=[1, 3, 5], max_err=1)
    reasoner = ReasonerBase(kb, "hamming")
    test_hwf(reasoner)

    print("HWF_KB without GKB, max_err=1")
    kb = HWF_KB(max_err=1)
    reasoner = ReasonerBase(kb, "hamming")
    test_hwf(reasoner)

    print("HWF_KB with multiple inputs at once:")
    kb = HWF_KB(max_err=0.1)
    reasoner = ReasonerBase(kb, "hamming")
    test_hwf_multiple(reasoner, max_revisions=[1,3,3])
    
    print("max_revision is float")
    test_hwf_multiple(reasoner, max_revisions=[0.5,0.9,0.9])

    class HED_prolog_KB(prolog_KB):
        def __init__(self, pseudo_label_list, pl_file):
            super().__init__(pseudo_label_list, pl_file)

        def consist_rule(self, exs, rules):
            rules = str(rules).replace("'", "")
            pl_query = "eval_inst_feature(%s, %s)." % (exs, rules)
            return len(list(self.prolog.query(pl_query))) != 0

        def abduce_rules(self, pred_res):
            pl_query = "consistent_inst_feature(%s, X)." % pred_res
            prolog_result = list(self.prolog.query(pl_query))
            if len(prolog_result) == 0:
                return None
            prolog_rules = prolog_result[0]["X"]
            rules = [rule.value for rule in prolog_rules]
            return rules

    class HED_Reasoner(ReasonerBase):
        def __init__(self, kb, dist_func="hamming"):
            super().__init__(kb, dist_func, use_zoopt=True)

        def _revise_at_idxs(self, pred_res, y, all_revision_flag, idxs):
            pred = []
            k = []
            revision_flag = []
            for idx in idxs:
                pred.append(pred_res[idx])
                k.append(y[idx])
                revision_flag += list(all_revision_flag[idx])
            revision_idx = np.where(np.array(revision_flag) != 0)[0]
            candidate = self.revise_at_idx(pred, k, revision_idx)
            return candidate

        def zoopt_revision_score(self, symbol_num, pred_res, pred_prob, y, sol):
            all_revision_flag = reform_idx(sol.get_x(), pred_res)
            lefted_idxs = [i for i in range(len(pred_res))]
            candidate_size = []
            while lefted_idxs:
                idxs = []
                idxs.append(lefted_idxs.pop(0))
                max_candidate_idxs = []
                found = False
                for idx in range(-1, len(pred_res)):
                    if (not idx in idxs) and (idx >= 0):
                        idxs.append(idx)
                    candidate = self._revise_at_idxs(
                        pred_res, y, all_revision_flag, idxs
                    )
                    if len(candidate) == 0:
                        if len(idxs) > 1:
                            idxs.pop()
                    else:
                        if len(idxs) > len(max_candidate_idxs):
                            found = True
                            max_candidate_idxs = idxs.copy()
                removed = [i for i in lefted_idxs if i in max_candidate_idxs]
                if found:
                    candidate_size.append(len(removed) + 1)
                    lefted_idxs = [
                        i for i in lefted_idxs if i not in max_candidate_idxs
                    ]
            candidate_size.sort()
            score = 0
            import math

            for i in range(0, len(candidate_size)):
                score -= math.exp(-i) * candidate_size[i]
            return score

        def abduce_rules(self, pred_res):
            return self.kb.abduce_rules(pred_res)

    kb = HED_prolog_KB(
        pseudo_label_list=[1, 0, "+", "="],
        pl_file="examples/hed/datasets/learn_add.pl",
    )
    reasoner = HED_Reasoner(kb)
    consist_exs = [
        [1, 1, "+", 0, "=", 1, 1],
        [1, "+", 1, "=", 1, 0],
        [0, "+", 0, "=", 0],
    ]
    inconsist_exs1 = [
        [1, 1, "+", 0, "=", 1, 1],
        [1, "+", 1, "=", 1, 0],
        [0, "+", 0, "=", 0],
        [0, "+", 0, "=", 1],
    ]
    inconsist_exs2 = [[1, "+", 0, "=", 0], [1, "=", 1, "=", 0], [0, "=", 0, "=", 1, 1]]
    rules = ["my_op([0], [0], [0])", "my_op([1], [1], [1, 0])"]

    print("HED_kb logic forward")
    print(kb.logic_forward(consist_exs))
    print(kb.logic_forward(inconsist_exs1), kb.logic_forward(inconsist_exs2))
    print()
    print("HED_kb consist rule")
    print(kb.consist_rule([1, "+", 1, "=", 1, 0], rules))
    print(kb.consist_rule([1, "+", 1, "=", 1, 1], rules))
    print()

    print("HED_Reasoner abduce")
    res = reasoner.abduce(
        [[[None]]] * len(consist_exs), consist_exs, [None] * len(consist_exs)
    )
    print(res)
    res = reasoner.abduce(
        [[[None]]] * len(inconsist_exs1), inconsist_exs1, [None] * len(inconsist_exs1)
    )
    print(res)
    res = reasoner.abduce(
        [[[None]]] * len(inconsist_exs2), inconsist_exs2, [None] * len(inconsist_exs2)
    )
    print(res)
    print()

    print("HED_Reasoner abduce rules")
    abduced_rules = reasoner.abduce_rules(consist_exs)
    print(abduced_rules)
