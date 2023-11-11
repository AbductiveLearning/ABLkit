# coding: utf-8
# ================================================================#
#   Copyright (C) 2020 Freecss All rights reserved.
#
#   File Name     ：models.py
#   Author        ：freecss
#   Email         ：karlfreecss@gmail.com
#   Created Date  ：2020/04/02
#   Description   ：
#
# ================================================================#
import pickle
from typing import Any, Dict

from ..structures import ListData
from ..utils import reform_idx


class ABLModel:
    """
    Serialize data and provide a unified interface for different machine learning models.

    Parameters
    ----------
    base_model : Machine Learning Model
        The base model to use for training and prediction.

    Attributes
    ----------
    classifier_list : List[Any]
        A list of classifiers.

    Methods
    -------
    predict(X: List[List[Any]], mapping: Optional[Dict] = None) -> Dict
        Predict the labels and probabilities for the given data.
    valid(X: List[List[Any]], Y: List[Any]) -> float
        Calculate the accuracy score for the given data.
    train(X: List[List[Any]], Y: List[Any]) -> float
        Train the model on the given data.
    save(*args, **kwargs) -> None
        Save the model to a file.
    load(*args, **kwargs) -> None
        Load the model from a file.
    """

    def __init__(self, base_model: Any) -> None:
        if not (hasattr(base_model, "fit") and hasattr(base_model, "predict")):
            raise NotImplementedError("The base_model should implement fit and predict methods.")

        self.base_model = base_model

    def predict(self, data_samples: ListData) -> Dict:
        """
        Predict the labels and probabilities for the given data.

        Parameters
        ----------
        X : List[List[Any]]
            The data to predict on.

        Returns
        -------
        dict
            A dictionary containing the predicted labels and probabilities.
        """
        model = self.base_model
        data_X = data_samples.flatten("X")
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X=data_X)
            label = prob.argmax(axis=1)
            prob = reform_idx(prob, data_samples["X"])
        else:
            prob = [None] * len(data_samples)
            label = model.predict(X=data_X)

        label = reform_idx(label, data_samples["X"])

        return {"label": label, "prob": prob}

    def train(self, data_samples: ListData) -> float:
        """
        Train the model on the given data.

        Parameters
        ----------
        X : List[List[Any]]
            The data to train on.
        Y : List[Any]
            The true labels for the given data.

        Returns
        -------
        float
            The loss value of the trained model.
        """
        data_X = data_samples.flatten("X")
        data_y = data_samples.flatten("abduced_idx")
        return self.base_model.fit(X=data_X, y=data_y)

    def _model_operation(self, operation: str, *args, **kwargs):
        model = self.base_model
        if hasattr(model, operation):
            method = getattr(model, operation)
            method(*args, **kwargs)
        else:
            if not f"{operation}_path" in kwargs.keys():
                raise ValueError(f"'{operation}_path' should not be None")
            else:
                try:
                    if operation == "save":
                        with open(kwargs["save_path"], "wb") as file:
                            pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
                    elif operation == "load":
                        with open(kwargs["load_path"], "rb") as file:
                            self.base_model = pickle.load(file)
                except:
                    raise NotImplementedError(
                        f"{type(model).__name__} object doesn't have the {operation} method and the default pickle-based {operation} method failed."
                    )

    def save(self, *args, **kwargs) -> None:
        """
        Save the model to a file.
        """
        self._model_operation("save", *args, **kwargs)

    def load(self, *args, **kwargs) -> None:
        """
        Load the model from a file.
        """
        self._model_operation("load", *args, **kwargs)
