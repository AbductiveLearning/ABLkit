
from abl.reasoning import ReasonerBase, BaseKB, GroundKB, PrologBasedKB

if __name__ == "__main__":
    prob1 = [
        [
            [0, 0.99, 0.01, 0, 0, 0, 0, 0, 0, 0],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ]
    ]

    prob2 = [
        [
            [0, 0, 0.01, 0, 0, 0, 0, 0.99, 0, 0],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ]
    ]

    class add_KB(BaseKB):
        def __init__(self, pseudo_label_list=list(range(10)), use_cache=True):
            super().__init__(pseudo_label_list, use_cache=use_cache)

        def logic_forward(self, nums):
            return sum(nums)

    class add_GroundKB(GroundKB):
        def __init__(self, pseudo_label_list=list(range(10)), GKB_len_list=[2]):
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
    kb = add_GroundKB()
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

    print("PrologBasedKB with add.pl:")
    kb = PrologBasedKB(
        pseudo_label_list=list(range(10)), pl_file="examples/mnist_add/datasets/add.pl"
    )
    reasoner = ReasonerBase(kb, "confidence")
    test_add(reasoner)

    print("PrologBasedKB with add.pl using zoopt:")
    kb = PrologBasedKB(
        pseudo_label_list=list(range(10)),
        pl_file="examples/mnist_add/datasets/add.pl",
    )
    reasoner = ReasonerBase(kb, "confidence", use_zoopt=True)
    test_add(reasoner)

    print("add_KB with multiple inputs at once:")
    multiple_prob = [
        [
            [0, 0.99, 0.01, 0, 0, 0, 0, 0, 0, 0],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ],
        [
            [0, 0, 0.01, 0, 0, 0, 0, 0.99, 0, 0],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ],
    ]

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

    class HWF_KB(BaseKB):
        def __init__(
            self,
            pseudo_label_list=[
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "+",
                "-",
                "times",
                "div",
            ],
            max_err=1e-3,
        ):
            super().__init__(pseudo_label_list, max_err)

        def _valid_candidate(self, formula):
            if len(formula) % 2 == 0:
                return False
            for i in range(len(formula)):
                if i % 2 == 0 and formula[i] not in [
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    "7",
                    "8",
                    "9",
                ]:
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

    class HWF_GroundKB(GroundKB):
        def __init__(
            self,
            pseudo_label_list=[
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "+",
                "-",
                "times",
                "div",
            ],
            GKB_len_list=[1, 3, 5, 7],
            max_err=1e-3,
        ):
            super().__init__(pseudo_label_list, GKB_len_list, max_err)

        def _valid_candidate(self, formula):
            if len(formula) % 2 == 0:
                return False
            for i in range(len(formula)):
                if i % 2 == 0 and formula[i] not in [
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    "7",
                    "8",
                    "9",
                ]:
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
    kb = HWF_GroundKB(GKB_len_list=[1, 3, 5], max_err=0.1)
    reasoner = ReasonerBase(kb, "hamming")
    test_hwf(reasoner)

    print("HWF_KB without GKB, max_err=0.1")
    kb = HWF_KB(max_err=0.1)
    reasoner = ReasonerBase(kb, "hamming")
    test_hwf(reasoner)

    print("HWF_KB with GKB, max_err=1")
    kb = HWF_GroundKB(GKB_len_list=[1, 3, 5], max_err=1)
    reasoner = ReasonerBase(kb, "hamming")
    test_hwf(reasoner)

    print("HWF_KB without GKB, max_err=1")
    kb = HWF_KB(max_err=1)
    reasoner = ReasonerBase(kb, "hamming")
    test_hwf(reasoner)

    print("HWF_KB with multiple inputs at once:")
    kb = HWF_KB(max_err=0.1)
    reasoner = ReasonerBase(kb, "hamming")
    test_hwf_multiple(reasoner, max_revisions=[1, 3, 3])

    print("max_revision is float")
    test_hwf_multiple(reasoner, max_revisions=[0.5, 0.9, 0.9])

    class HED_prolog_KB(PrologBasedKB):
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
                    candidate = self._revise_at_idxs(pred_res, y, all_revision_flag, idxs)
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
                    lefted_idxs = [i for i in lefted_idxs if i not in max_candidate_idxs]
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
    res = reasoner.abduce([[[None]]] * len(consist_exs), consist_exs, [None] * len(consist_exs))
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
