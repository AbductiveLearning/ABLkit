from unittest.mock import Mock, create_autospec

import numpy as np
import pytest

from abl.learning import ABLModel


class TestABLModel(object):
    def test_ablmodel_initialization(self):
        """Test the initialization method of the ABLModel class."""

        invalid_base_model = Mock(spec=[])

        with pytest.raises(NotImplementedError):
            ABLModel(invalid_base_model)

        fit = Mock(return_value=1.0)
        predict = Mock(return_value=np.array(1.0))

        invalid_base_model = Mock(spec=fit)
        with pytest.raises(NotImplementedError):
            ABLModel(invalid_base_model)

        invalid_base_model = Mock(spec=predict)
        with pytest.raises(NotImplementedError):
            ABLModel(invalid_base_model)

        base_model = Mock(spec=["fit", "predict"])
        base_model.fit = fit
        base_model.predict = predict
        model = ABLModel(base_model)
        assert hasattr(model, "base_model"), "The model should have a 'base_model' attribute."

    def test_ablmodel_predict(self, base_model_instance, list_data_instance):
        """Test the predict method of the ABLModel class."""
        model = ABLModel(base_model_instance)
        predictions = model.predict(list_data_instance)
        assert isinstance(predictions, dict), "Predictions should be returned as a dictionary."
        assert isinstance(predictions["label"], list)
        assert isinstance(predictions["prob"], list)

        basic_nn_mock = create_autospec(base_model_instance, spec_set=True)
        delattr(basic_nn_mock, "predict_proba")
        model = ABLModel(basic_nn_mock)
        predictions = model.predict(list_data_instance)
        assert isinstance(predictions, dict), "Predictions should be returned as a dictionary."
        assert isinstance(predictions["label"], list)
        assert predictions["prob"] is None

    def test_ablmodel_train(self, base_model_instance, list_data_instance):
        """Test the train method of the ABLModel class."""
        model = ABLModel(base_model_instance)
        list_data_instance.abduced_idx = [[1, 2], [3, 4], [5, 6]]
        ins = model.train(list_data_instance)
        assert ins == model.base_model, "Training should return the base model instance."

    def test_ablmodel_save_load(self, base_model_instance, tmp_path):
        """Test the save method of the ABLModel class."""
        model = ABLModel(base_model_instance)
        model_path = tmp_path / "model.pth"
        model.save(save_path=str(model_path))
        assert model_path.exists()
        model.load(load_path=str(model_path))
        assert isinstance(model.base_model, type(base_model_instance))

    def test_ablmodel_invalid_operation(self, base_model_instance):
        """Test invalid operation handling in the ABLModel class."""
        model = ABLModel(base_model_instance)
        with pytest.raises(ValueError):
            model._model_operation("invalid_operation", save_path=None)

    def test_ablmodel_operation_without_path(self, base_model_instance):
        """Test operation without providing a path in the ABLModel class."""
        model = ABLModel(base_model_instance)
        with pytest.raises(ValueError):
            model.save()  # No path provided
