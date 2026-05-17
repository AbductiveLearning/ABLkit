import logging
import pickle
from typing import Any, Dict, List, Optional

import numpy
import torch
from torch.utils.data import DataLoader

from ablkit.data import ListData
from ablkit.learning import ABLModel
from ablkit.learning.basic_nn import BasicNN
from ablkit.learning.torch_dataset import PredictionDataset
from ablkit.utils import print_log
from ablkit.utils.utils import reform_list

class A3BLBasicNN(BasicNN):
    def _extract_features(self, data_loader: DataLoader) -> torch.Tensor:
        """
        Internal method to predict the outputs given a DataLoader.

        Parameters
        ----------
        data_loader : DataLoader
            The DataLoader providing input samples.

        Returns
        -------
        torch.Tensor
            Raw output from the model.
        """
        if not isinstance(data_loader, DataLoader):
            raise TypeError(
                f"data_loader must be an instance of torch.utils.data.DataLoader, "
                f"but got {type(data_loader)}"
            )
        model = self.model
        device = self.device

        model.eval()

        with torch.no_grad():
            results = []
            for data in data_loader:
                data = data.to(device)
                out = model.extract_features(data)
                results.append(out)

        return torch.cat(results, axis=0)

    def extract_features(
        self,
        data_loader: Optional[DataLoader] = None,
        X: Optional[List[Any]] = None,
    ) -> numpy.ndarray:
        """
        Predict the class of the input data. This method supports prediction with either
        a DataLoader object (data_loader) or a list of input data (X). If both data_loader
        and X are provided, the method will predict the input data in data_loader
        instead of X.

        Parameters
        ----------
        data_loader : DataLoader, optional
            The data loader used for prediction. Defaults to None.
        X : List[Any], optional
            The input data. Defaults to None.

        Returns
        -------
        numpy.ndarray
            The predicted class of the input data.
        """

        if data_loader is not None and X is not None:
            print_log(
                "Extract the feature of input data in data_loader instead of X.",
                logger="current",
                level=logging.WARNING,
            )

        if data_loader is None:
            dataset = PredictionDataset(X, self.test_transform)
            data_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                pin_memory=torch.cuda.is_available(),
            )
        return self._extract_features(data_loader).cpu().numpy()

class A3BLModel(ABLModel):
    """
    Serialize data and provide a unified interface for different machine learning models.

    Parameters
    ----------
    base_model : Machine Learning Model
        The machine learning base model used for training and prediction. This model should
        implement the ``fit`` and ``predict`` methods. It's recommended, but not required, for the
        model to also implement the ``predict_proba`` method for generating
        predictions on the probabilities.
    """

    def __init__(self, base_model):
        super().__init__(base_model)

    def predict(self, data_examples: ListData) -> Dict:
        """
        Predict the labels and probabilities for the given data.

        Parameters
        ----------
        data_examples : ListData
            A batch of data to predict on.

        Returns
        -------
        dict
            A dictionary containing the predicted labels and probabilities.
        """
        model = self.base_model
        data_X = data_examples.flatten("X")
        embeddings = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X=data_X)
            if hasattr(model, "extract_features"):
                embeddings = model.extract_features(X=data_X)
            label = prob.argmax(axis=1)
            prob = reform_list(prob, data_examples.X)
        else:
            prob = None
            label = model.predict(X=data_X)
        label = reform_list(label, data_examples.X)

        data_examples.pred_idx = label
        data_examples.pred_prob = prob
        if embeddings is not None:
            data_examples.embeddings = reform_list(embeddings, data_examples.X)

        return {"label": label, "prob": prob}

    def train(self, data_examples: ListData) -> float:
        """
        Train the model on the given data.

        Parameters
        ----------
        data_examples : ListData
            A batch of data to train on, which typically contains the data, ``X``, and the
            corresponding labels, ``abduced_idx``.

        Returns
        -------
        float
            The loss value of the trained model.
        """
        data_X = data_examples.flatten("X")
        data_y = data_examples.flatten("abduced_idx")
        return self.base_model.fit(X=data_X, y=data_y)

    def valid(self, data_examples: ListData) -> float:
        """
        Validate the model on the given data.

        Parameters
        ----------
        data_examples : ListData
            A batch of data to train on, which typically contains the data, ``X``,
            and the corresponding labels, ``abduced_idx``.

        Returns
        -------
        float
            The accuracy of the trained model.
        """
        data_X = data_examples.flatten("X")
        data_y = data_examples.flatten("abduced_idx")
        score = self.base_model.score(X=data_X, y=data_y)
        return score

    def _model_operation(self, operation: str, *args, **kwargs):
        model = self.base_model
        if hasattr(model, operation):
            method = getattr(model, operation)
            method(*args, **kwargs)
        else:
            if f"{operation}_path" not in kwargs:
                raise ValueError(f"'{operation}_path' should not be None")
            try:
                if operation == "save":
                    with open(kwargs["save_path"], "wb") as file:
                        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
                elif operation == "load":
                    with open(kwargs["load_path"], "rb") as file:
                        self.base_model = pickle.load(file)
            except (OSError, pickle.PickleError) as exc:
                raise NotImplementedError(
                    f"{type(model).__name__} object doesn't have the {operation} method \
                        and the default pickle-based {operation} method failed."
                ) from exc

    def save(self, *args, **kwargs) -> None:
        """
        Save the model to a file.

        This method delegates to the ``save`` method of self.base_model. The arguments passed to
        this method should match those expected by the ``save`` method of self.base_model.
        """
        self._model_operation("save", *args, **kwargs)

    def load(self, *args, **kwargs) -> None:
        """
        Load the model from a file.

        This method delegates to the ``load`` method of self.base_model. The arguments passed to
        this method should match those expected by the ``load`` method of self.base_model.
        """
        self._model_operation("load", *args, **kwargs)
