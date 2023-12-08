import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abl.learning import BasicNN
from abl.reasoning import GroundKB, KBBase, PrologKB, Reasoner
from abl.structures import ListData
from examples.models.nn import LeNet5


# Fixture for BasicNN instance
@pytest.fixture
def basic_nn_instance():
    model = LeNet5()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    return BasicNN(model, criterion, optimizer)


# Fixture for base_model instance
@pytest.fixture
def base_model_instance():
    model = LeNet5()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    return BasicNN(model, criterion, optimizer)


# Fixture for ListData instance
@pytest.fixture
def list_data_instance():
    data_samples = ListData()
    data_samples.X = [list(torch.randn(2, 1, 28, 28)) for _ in range(3)]
    data_samples.Y = [1, 2, 3]
    data_samples.gt_pseudo_label = [[1, 2], [3, 4], [5, 6]]
    return data_samples


@pytest.fixture
def data_samples_add():
    # favor 1 in first one
    prob1 = [
        [0, 0.99, 0, 0, 0, 0, 0, 0.01, 0, 0],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    ]
    # favor 7 in first one
    prob2 = [
        [0, 0.01, 0, 0, 0, 0, 0, 0.99, 0, 0],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    ]

    data_samples_add = ListData()
    data_samples_add.pred_pseudo_label = [[1, 1], [1, 1], [1, 1], [1, 1]]
    data_samples_add.pred_prob = [prob1, prob2, prob1, prob2]
    data_samples_add.Y = [8, 8, 17, 10]
    return data_samples_add


@pytest.fixture
def data_samples_hwf():
    data_samples_hwf = ListData()
    data_samples_hwf.pred_pseudo_label = [
        ["5", "+", "2"],
        ["5", "+", "9"],
        ["5", "+", "9"],
        ["5", "-", "8", "8", "8"],
    ]
    data_samples_hwf.pred_prob = [None, None, None, None]
    data_samples_hwf.Y = [3, 64, 65, 3.17]
    return data_samples_hwf


class AddKB(KBBase):
    def __init__(self, pseudo_label_list=list(range(10)), use_cache=False):
        super().__init__(pseudo_label_list, use_cache=use_cache)

    def logic_forward(self, nums):
        return sum(nums)


class AddGroundKB(GroundKB):
    def __init__(self, pseudo_label_list=list(range(10)), GKB_len_list=[2]):
        super().__init__(pseudo_label_list, GKB_len_list)

    def logic_forward(self, nums):
        return sum(nums)


class HwfKB(KBBase):
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
        use_cache=False,
    ):
        super().__init__(pseudo_label_list, max_err, use_cache)

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
            return None
        mapping = {str(i): str(i) for i in range(1, 10)}
        mapping.update({"+": "+", "-": "-", "times": "*", "div": "/"})
        formula = [mapping[f] for f in formula]
        return eval("".join(formula))


class HedKB(PrologKB):
    def __init__(self, pseudo_label_list, pl_file):
        super().__init__(pseudo_label_list, pl_file)

    def consist_rule(self, exs, rules):
        rules = str(rules).replace("'", "")
        pl_query = "eval_inst_feature(%s, %s)." % (exs, rules)
        return len(list(self.prolog.query(pl_query))) != 0


@pytest.fixture
def kb_add():
    return AddKB()


@pytest.fixture
def kb_add_cache():
    return AddKB(use_cache=True)


@pytest.fixture
def kb_add_ground():
    return AddGroundKB()


@pytest.fixture
def kb_add_prolog():
    kb = PrologKB(pseudo_label_list=list(range(10)), pl_file="examples/mnist_add/datasets/add.pl")
    return kb


@pytest.fixture
def kb_hed():
    kb = HedKB(
        pseudo_label_list=[1, 0, "+", "="],
        pl_file="examples/hed/datasets/learn_add.pl",
    )
    return kb


@pytest.fixture
def reasoner_instance(kb_add):
    return Reasoner(kb_add, "confidence")
