import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from abl.learning import BasicNN
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
