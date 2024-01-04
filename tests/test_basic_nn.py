import numpy
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class TestBasicNN(object):
    @pytest.fixture
    def sample_data(self):
        return torch.randn(32, 1, 28, 28), torch.randint(0, 10, (32,))

    @pytest.fixture
    def sample_data_loader_with_label(self, sample_data):
        X, y = sample_data
        return DataLoader(TensorDataset(X, y), batch_size=4)

    @pytest.fixture
    def sample_data_loader_without_label(self, sample_data):
        X, _ = sample_data
        return DataLoader(
            TensorDataset(X),
            batch_size=4,
            collate_fn=lambda batch: torch.stack([item[0] for item in batch]),
        )

    def test_initialization(self, basic_nn_instance):
        """Test initialization of the BasicNN class"""
        assert basic_nn_instance.model is not None
        assert isinstance(basic_nn_instance.loss_fn, nn.Module)
        assert isinstance(basic_nn_instance.optimizer, optim.Optimizer)

    def test_training_methods(self, basic_nn_instance, sample_data, sample_data_loader_with_label):
        """Test train_epoch, fit, and score methods of the BasicNN class"""

        # Test train_epoch
        loss = basic_nn_instance.train_epoch(sample_data_loader_with_label)
        assert isinstance(loss, float)

        # Test fit with direct data
        X, y = sample_data
        ins = basic_nn_instance.fit(X=list(X), y=list(y))
        assert ins == basic_nn_instance

        # Test fit with DataLoader
        ins = basic_nn_instance.fit(data_loader=sample_data_loader_with_label)
        assert ins == basic_nn_instance

        # Test invalid fit method input
        with pytest.raises(ValueError):
            basic_nn_instance.fit(X=None, y=None, data_loader=None)

        # Test score with direct data
        accuracy = basic_nn_instance.score(X=list(X), y=list(y))
        assert 0 <= accuracy <= 1

        # Test score with DataLoader
        accuracy = basic_nn_instance.score(data_loader=sample_data_loader_with_label)
        assert 0 <= accuracy <= 1

    def test_prediction_methods(
        self, basic_nn_instance, sample_data, sample_data_loader_without_label
    ):
        """Test predict and predict_proba methods of the BasicNN class"""
        X, _ = sample_data

        # Test predict with direct data
        predictions = basic_nn_instance.predict(X=list(X))
        assert len(predictions) == len(X)
        assert numpy.isin(predictions, list(range(10))).all()

        # Test predict_proba with direct data
        predict_proba = basic_nn_instance.predict_proba(X=list(X))
        assert len(predict_proba) == len(X)
        assert ((0 <= predict_proba) & (predict_proba <= 1)).all()

        # Test predict and predict_proba with DataLoader
        for method in [basic_nn_instance.predict, basic_nn_instance.predict_proba]:
            result = method(data_loader=sample_data_loader_without_label)
            assert len(result) == len(X)
            if method == basic_nn_instance.predict:
                assert numpy.isin(result, list(range(10))).all()
            else:
                assert ((0 <= result) & (result <= 1)).all()

    def test_save_load(self, basic_nn_instance, tmp_path):
        """Test save and load methods of the BasicNN class"""

        # Test save with explicit save_path
        explicit_save_path = tmp_path / "model_explicit.pth"
        basic_nn_instance.save(epoch_id=1, save_path=str(explicit_save_path))
        assert explicit_save_path.exists(), "Model should be saved to the explicit path"

        # Test save without providing save_path (using save_dir)
        basic_nn_instance.save_dir = str(tmp_path)
        implicit_save_path = tmp_path / "model_checkpoint_epoch_1.pth"
        basic_nn_instance.save(epoch_id=1)
        assert implicit_save_path.exists(), "Model should be saved to the implicit path in save_dir"

        # Test error when save_path and save_dir are both None
        basic_nn_instance.save_dir = None
        with pytest.raises(ValueError):
            basic_nn_instance.save(epoch_id=1)

        # Test error on loading from a None path
        with pytest.raises(ValueError):
            basic_nn_instance.load(load_path=None)

        # Test loading model state
        original_state = basic_nn_instance.model.state_dict()
        basic_nn_instance.load(load_path=str(explicit_save_path))
        loaded_state = basic_nn_instance.model.state_dict()
        for key in original_state:
            assert torch.allclose(
                original_state[key], loaded_state[key]
            ), "Model state should be restored after loading"
