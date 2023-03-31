import sys

sys.path.insert(0, sys.path[0] + "/../")

import os
import pytest
import torch
import torch.nn as nn
import numpy as np

from abl.models.nn import LeNet5, SymbolNet
from abl.models.basic_model import BasicModel


class TestBasicModel(object):
    @pytest.mark.parametrize("num_classes", [4, 10])
    @pytest.mark.parametrize("image_size", [(28, 28, 1), (45, 45, 1)])
    @pytest.mark.parametrize("cls", [LeNet5, SymbolNet])
    @pytest.mark.parametrize("criterion", [nn.CrossEntropyLoss])
    @pytest.mark.parametrize("optimizer", [torch.optim.RMSprop])
    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_models(self, num_classes, image_size, cls, criterion, optimizer, device):
        cls = cls(num_classes=num_classes, image_size=image_size)
        criterion = criterion()
        optimizer = optimizer(cls.parameters(), lr=0.001)

        self.num_classes = num_classes
        self.image_size = image_size
        self.model = BasicModel(cls, criterion, optimizer, device)

        self.data_X = [
            np.random.rand(image_size[2], image_size[0], image_size[1]).astype(
                np.float32
            )
            for i in range(5)
        ]
        self.data_y = np.random.randint(0, num_classes, (5,))

        self._test_fit()
        self._test_predict()
        self._test_predict_proba()
        self._test_score()
        self._test_save()
        self._test_load()

    def _test_fit(self):
        self.model.fit(X=self.data_X, y=self.data_y)

    def _test_predict(self):
        predict_result = self.model.predict(X=self.data_X)
        assert predict_result.dtype == int
        assert predict_result.shape == (5,)
        assert (0 <= predict_result).all() and (predict_result < self.num_classes).all()

    def _test_predict_proba(self):
        predict_result = self.model.predict_proba(X=self.data_X)
        assert predict_result.dtype == np.float32
        assert predict_result.shape == (5, self.num_classes)
        assert (0 <= predict_result).all() and (predict_result <= 1).all()

    def _test_score(self):
        accuracy = self.model.score(X=self.data_X, y=self.data_y)
        assert type(accuracy) == float
        assert 0 <= accuracy <= 1

    def _test_save(self):
        self.model.save(1, "results/test_models")
        assert os.path.exists("results/test_models/1_net.pth")
        assert os.path.exists("results/test_models/1_opt.pth")
        os.remove("results/test_models/1_net.pth")
        os.remove("results/test_models/1_opt.pth")

    def _test_load(self):
        self.model.save(1, "results/test_models")
        self.model.load(1, "results/test_models")
