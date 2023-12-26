import numpy as np
import pytest

from abl.reasoning import PrologKB, Reasoner


class TestKBBase(object):
    def test_init(self, kb_add):
        assert kb_add.pseudo_label_list == list(range(10))

    def test_init_cache(self, kb_add_cache):
        assert kb_add_cache.pseudo_label_list == list(range(10))
        assert kb_add_cache.use_cache is True

    def test_logic_forward(self, kb_add):
        result = kb_add.logic_forward([1, 2])
        assert result == 3
        with pytest.raises(TypeError):
            kb_add.logic_forward([1, 2], [0.1, -0.2, 0.2, -0.3])

    def test_revise_at_idx(self, kb_add):
        result = kb_add.revise_at_idx([0, 2], 2, [0.1, -0.2, 0.2, -0.3], [])
        assert result == ([[0, 2]], [2])
        result = kb_add.revise_at_idx([1, 2], 2, [0.1, -0.2, 0.2, -0.3], [])
        assert result == ([], [])
        result = kb_add.revise_at_idx([1, 2], 2, [0.1, -0.2, 0.2, -0.3], [0, 1])
        assert result == ([[0, 2], [1, 1], [2, 0]], [2, 2, 2])

    def test_abduce_candidates(self, kb_add):
        result = kb_add.abduce_candidates([0, 1], 1, [0.1, -0.2, 0.2, -0.3], max_revision_num=2, require_more_revision=0)
        assert result == ([[0, 1]], [1])
        result = kb_add.abduce_candidates([1, 2], 1, [0.1, -0.2, 0.2, -0.3], max_revision_num=2, require_more_revision=0)
        assert result == ([[1, 0]], [1])


class TestGroundKB(object):
    def test_init(self, kb_add_ground):
        assert kb_add_ground.pseudo_label_list == list(range(10))
        assert kb_add_ground.GKB_len_list == [2]
        assert kb_add_ground.GKB

    def test_logic_forward_ground(self, kb_add_ground):
        result = kb_add_ground.logic_forward([1, 2])
        assert result == 3

    def test_abduce_candidates_ground(self, kb_add_ground):
        result = kb_add_ground.abduce_candidates(
            [1, 2], 1, [0.1, -0.2, 0.2, -0.3], max_revision_num=2, require_more_revision=0
        )
        assert result == ([(1, 0)], [1])


class TestPrologKB(object):
    def test_init_pl1(self, kb_add_prolog):
        assert kb_add_prolog.pseudo_label_list == list(range(10))
        assert kb_add_prolog.pl_file == "examples/mnist_add/add.pl"

    def test_init_pl2(self, kb_hed):
        assert kb_hed.pseudo_label_list == [1, 0, "+", "="]
        assert kb_hed.pl_file == "examples/hed/reasoning/learn_add.pl"

    def test_prolog_file_not_exist(self):
        pseudo_label_list = [1, 2]
        non_existing_file = "path/to/non_existing_file.pl"
        with pytest.raises(FileNotFoundError) as excinfo:
            PrologKB(pseudo_label_list=pseudo_label_list, pl_file=non_existing_file)
        assert non_existing_file in str(excinfo.value)

    def test_logic_forward_pl1(self, kb_add_prolog):
        result = kb_add_prolog.logic_forward([1, 2])
        assert result == 3

    def test_logic_forward_pl2(self, kb_hed):
        consist_exs = [
            [1, 1, "+", 0, "=", 1, 1],
            [1, "+", 1, "=", 1, 0],
            [0, "+", 0, "=", 0],
        ]
        inconsist_exs = [
            [1, 1, "+", 0, "=", 1, 1],
            [1, "+", 1, "=", 1, 0],
            [0, "+", 0, "=", 0],
            [0, "+", 0, "=", 1],
        ]
        assert kb_hed.logic_forward(consist_exs) is True
        assert kb_hed.logic_forward(inconsist_exs) is False

    def test_revise_at_idx(self, kb_add_prolog):
        result = kb_add_prolog.revise_at_idx([1, 2], 2, [0.1, -0.2, 0.2, -0.3], [0])
        assert result == ([[0, 2]], [2])


class TestReaonser(object):
    def test_reasoner_init(self, reasoner_instance):
        assert reasoner_instance.dist_func == "confidence"

    def test_invalid_predefined_dist_func(self, kb_add):
        with pytest.raises(NotImplementedError) as excinfo:
            Reasoner(kb_add, "invalid_dist_func")
        assert 'Valid options for predefined dist_func include "hamming" and "confidence"' in str(
            excinfo.value
        )
        
    def random_dist(self, data_example, candidates, candidate_idxs, reasoning_results):
        cost_list = [np.random.rand() for _ in candidates]
        return cost_list
    
    def test_user_defined_dist_func(self, kb_add):
        reasoner = Reasoner(kb_add, self.random_dist)
        assert reasoner.dist_func == self.random_dist
    
    def invalid_dist1(self, candidates):
        cost_list = np.array([np.random.rand() for _ in candidates])
        return cost_list
    
    def invalid_dist2(self, data_example, candidates, candidate_idxs, reasoning_results):
        cost_list = np.array([np.random.rand() for _ in candidates])
        return np.append(cost_list, np.random.rand())
    
    def test_invalid_user_defined_dist_func(self, kb_add, data_examples_add):
        with pytest.raises(ValueError) as excinfo:
            Reasoner(kb_add, self.invalid_dist1)
        assert 'User-defined dist_func must have exactly four parameters' in str(
            excinfo.value
        )
        with pytest.raises(ValueError) as excinfo:
            reasoner = Reasoner(kb_add, self.invalid_dist2)
            reasoner.batch_abduce(data_examples_add)
        assert 'The length of the array returned by dist_func must be equal to the number of candidates' in str(
            excinfo.value
        )


class TestBatchAbduce(object):
    def test_batch_abduce_add(self, kb_add, data_examples_add):
        reasoner1 = Reasoner(kb_add, "confidence", max_revision=1, require_more_revision=0)
        reasoner2 = Reasoner(kb_add, "confidence", max_revision=1, require_more_revision=1)
        reasoner3 = Reasoner(kb_add, "confidence", max_revision=2, require_more_revision=0)
        reasoner4 = Reasoner(kb_add, "confidence", max_revision=2, require_more_revision=1)
        assert reasoner1.batch_abduce(data_examples_add) == [[1, 7], [7, 1], [], [1, 9]]
        assert reasoner2.batch_abduce(data_examples_add) == [[1, 7], [7, 1], [], [1, 9]]
        assert reasoner3.batch_abduce(data_examples_add) == [
            [1, 7],
            [7, 1],
            [8, 9],
            [1, 9],
        ]
        assert reasoner4.batch_abduce(data_examples_add) == [
            [1, 7],
            [7, 1],
            [8, 9],
            [7, 3],
        ]

    def test_batch_abduce_ground(self, kb_add_ground, data_examples_add):
        reasoner1 = Reasoner(kb_add_ground, "confidence", max_revision=1, require_more_revision=0)
        reasoner2 = Reasoner(kb_add_ground, "confidence", max_revision=1, require_more_revision=1)
        reasoner3 = Reasoner(kb_add_ground, "confidence", max_revision=2, require_more_revision=0)
        reasoner4 = Reasoner(kb_add_ground, "confidence", max_revision=2, require_more_revision=1)
        assert reasoner1.batch_abduce(data_examples_add) == [(1, 7), (7, 1), [], (1, 9)]
        assert reasoner2.batch_abduce(data_examples_add) == [(1, 7), (7, 1), [], (1, 9)]
        assert reasoner3.batch_abduce(data_examples_add) == [
            (1, 7),
            (7, 1),
            (8, 9),
            (1, 9),
        ]
        assert reasoner4.batch_abduce(data_examples_add) == [
            (1, 7),
            (7, 1),
            (8, 9),
            (7, 3),
        ]

    def test_batch_abduce_prolog(self, kb_add_prolog, data_examples_add):
        reasoner1 = Reasoner(kb_add_prolog, "confidence", max_revision=1, require_more_revision=0)
        reasoner2 = Reasoner(kb_add_prolog, "confidence", max_revision=1, require_more_revision=1)
        reasoner3 = Reasoner(kb_add_prolog, "confidence", max_revision=2, require_more_revision=0)
        reasoner4 = Reasoner(kb_add_prolog, "confidence", max_revision=2, require_more_revision=1)
        assert reasoner1.batch_abduce(data_examples_add) == [[1, 7], [7, 1], [], [1, 9]]
        assert reasoner2.batch_abduce(data_examples_add) == [[1, 7], [7, 1], [], [1, 9]]
        assert reasoner3.batch_abduce(data_examples_add) == [
            [1, 7],
            [7, 1],
            [8, 9],
            [1, 9],
        ]
        assert reasoner4.batch_abduce(data_examples_add) == [
            [1, 7],
            [7, 1],
            [8, 9],
            [7, 3],
        ]

    def test_batch_abduce_zoopt(self, kb_add_prolog, data_examples_add):
        reasoner1 = Reasoner(kb_add_prolog, "confidence", use_zoopt=True, max_revision=1)
        reasoner2 = Reasoner(kb_add_prolog, "confidence", use_zoopt=True, max_revision=2)
        assert reasoner1.batch_abduce(data_examples_add) == [[1, 7], [7, 1], [], [1, 9]]
        assert reasoner2.batch_abduce(data_examples_add) == [
            [1, 7],
            [7, 1],
            [8, 9],
            [7, 3],
        ]

    def test_batch_abduce_hwf1(self, kb_hwf1, data_examples_hwf):
        reasoner1 = Reasoner(kb_hwf1, "hamming", max_revision=3, require_more_revision=0)
        reasoner2 = Reasoner(kb_hwf1, "hamming", max_revision=0.5, require_more_revision=0)
        reasoner3 = Reasoner(kb_hwf1, "hamming", max_revision=0.9, require_more_revision=0)
        res = reasoner1.batch_abduce(data_examples_hwf)
        assert res == [
            ["1", "+", "2"],
            ["8", "times", "8"],
            [],
            ["4", "-", "6", "div", "8"],
        ]
        res = reasoner2.batch_abduce(data_examples_hwf)
        assert res == [["1", "+", "2"], [], [], []]
        res = reasoner3.batch_abduce(data_examples_hwf)
        assert res == [
            ["1", "+", "2"],
            ["8", "times", "8"],
            [],
            ["4", "-", "6", "div", "8"],
        ]

    def test_batch_abduce_hwf2(self, kb_hwf2, data_examples_hwf):
        reasoner1 = Reasoner(kb_hwf2, "hamming", max_revision=3, require_more_revision=0)
        reasoner2 = Reasoner(kb_hwf2, "hamming", max_revision=0.5, require_more_revision=0)
        reasoner3 = Reasoner(kb_hwf2, "hamming", max_revision=0.9, require_more_revision=0)
        res = reasoner1.batch_abduce(data_examples_hwf)
        assert res == [
            ["1", "+", "2"],
            ["7", "times", "9"],
            ["8", "times", "8"],
            ["5", "-", "8", "div", "8"],
        ]
        res = reasoner2.batch_abduce(data_examples_hwf)
        assert res == [
            ["1", "+", "2"],
            ["7", "times", "9"],
            [],
            ["5", "-", "8", "div", "8"],
        ]
        res = reasoner3.batch_abduce(data_examples_hwf)
        assert res == [
            ["1", "+", "2"],
            ["7", "times", "9"],
            ["8", "times", "8"],
            ["5", "-", "8", "div", "8"],
        ]
