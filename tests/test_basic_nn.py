import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class TestBasicNN(object):
    # Test initialization
    def test_initialization(self, basic_nn_instance):
        assert basic_nn_instance.model is not None
        assert isinstance(basic_nn_instance.criterion, nn.Module)
        assert isinstance(basic_nn_instance.optimizer, optim.Optimizer)

    # Test training epoch
    def test_train_epoch(self, basic_nn_instance):
        X = torch.randn(32, 1, 28, 28)
        y = torch.randint(0, 10, (32,))
        data_loader = DataLoader(TensorDataset(X, y), batch_size=4)
        loss = basic_nn_instance.train_epoch(data_loader)
        assert isinstance(loss, float)

    # Test fit method
    def test_fit(self, basic_nn_instance):
        X = torch.randn(32, 1, 28, 28)
        y = torch.randint(0, 10, (32,))
        data_loader = DataLoader(TensorDataset(X, y), batch_size=4)
        loss = basic_nn_instance.fit(data_loader)
        assert isinstance(loss, float)

    # Test predict method
    def test_predict(self, basic_nn_instance):
        X = list(torch.randn(32, 1, 28, 28))
        predictions = basic_nn_instance.predict(X=X)
        assert len(predictions) == len(X)
        assert numpy.isin(predictions, list(range(10))).all()

    # Test predict_proba method
    def test_predict_proba(self, basic_nn_instance):
        X = list(torch.randn(32, 1, 28, 28))
        predict_proba = basic_nn_instance.predict_proba(X=X)
        assert len(predict_proba) == len(X)
        assert ((0 <= predict_proba) & (predict_proba <= 1)).all()

    # Test score method
    def test_score(self, basic_nn_instance):
        X = torch.randn(32, 1, 28, 28)
        y = torch.randint(0, 10, (32,))
        data_loader = DataLoader(TensorDataset(X, y), batch_size=4)
        accuracy = basic_nn_instance.score(data_loader)
        assert 0 <= accuracy <= 1

    # Test save and load methods
    def test_save_load(self, basic_nn_instance, tmp_path):
        model_path = tmp_path / "model.pth"
        basic_nn_instance.save(epoch_id=1, save_path=str(model_path))
        assert model_path.exists()
        basic_nn_instance.load(load_path=str(model_path))
