import numpy as np
import platform
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from abl.data.structures import ListData
from abl.learning import BasicNN
from abl.reasoning import GroundKB, KBBase, PrologKB, Reasoner


class LeNet5(nn.Module):
    def __init__(self, num_classes=10, image_size=(28, 28)):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 3), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(nn.Conv2d(16, 16, 3), nn.ReLU())

        feature_map_size = (np.array(image_size) // 2 - 2) // 2 - 2
        num_features = 16 * feature_map_size[0] * feature_map_size[1]

        self.fc1 = nn.Sequential(nn.Linear(num_features, 120), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120, 84), nn.ReLU())
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# Fixture for BasicNN instance
@pytest.fixture
def basic_nn_instance():
    model = LeNet5()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    return BasicNN(model, loss_fn, optimizer)


# Fixture for base_model instance
@pytest.fixture
def base_model_instance():
    model = LeNet5()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    return BasicNN(model, loss_fn, optimizer)


# Fixture for ListData instance
@pytest.fixture
def list_data_instance():
    data_examples = ListData()
    data_examples.X = [list(torch.randn(2, 1, 28, 28)) for _ in range(3)]
    data_examples.Y = [1, 2, 3]
    data_examples.gt_pseudo_label = [[1, 2], [3, 4], [5, 6]]
    return data_examples


@pytest.fixture
def data_examples_add():
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

    data_examples_add = ListData()
    data_examples_add.X = None
    data_examples_add.pred_pseudo_label = [[1, 1], [1, 1], [1, 1], [1, 1]]
    data_examples_add.pred_prob = [prob1, prob2, prob1, prob2]
    data_examples_add.Y = [8, 8, 17, 10]
    return data_examples_add


@pytest.fixture
def data_examples_hwf():
    data_examples_hwf = ListData()
    data_examples_hwf.X = None
    data_examples_hwf.pred_pseudo_label = [
        ["5", "+", "2"],
        ["5", "+", "9"],
        ["5", "+", "9"],
        ["5", "-", "8", "8", "8"],
    ]
    data_examples_hwf.pred_prob = [None, None, None, None]
    data_examples_hwf.Y = [3, 64, 65, 3.17]
    return data_examples_hwf


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
    if platform.system() == "Darwin":
        return
    kb = PrologKB(pseudo_label_list=list(range(10)), pl_file="examples/mnist_add/add.pl")
    return kb


@pytest.fixture
def kb_hwf1():
    return HwfKB(max_err=0.1)


@pytest.fixture
def kb_hwf2():
    return HwfKB(max_err=1)


@pytest.fixture
def kb_hed():
    if platform.system() == "Darwin":
        return
    kb = HedKB(
        pseudo_label_list=[1, 0, "+", "="],
        pl_file="examples/hed/reasoning/learn_add.pl",
    )
    return kb


@pytest.fixture
def reasoner_instance(kb_add):
    return Reasoner(kb_add, "confidence")
