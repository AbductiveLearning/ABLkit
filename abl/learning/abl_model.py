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
from typing import List, Any, Optional

import pickle

from ..structures import ListData
from ..utils import flatten, reform_idx



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
    predict(X: List[List[Any]], mapping: Optional[dict] = None) -> dict
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

    def __init__(self, base_model) -> None:
        self.classifier_list = []
        self.classifier_list.append(base_model)

        if not (
            hasattr(base_model, "fit")
            and hasattr(base_model, "predict")
            and hasattr(base_model, "score")
        ):
            raise NotImplementedError(
                "base_model should have fit, predict and score methods."
            )

    def predict(self, data_samples: ListData, mapping: Optional[dict] = None) -> dict:
        """
        Predict the labels and probabilities for the given data.

        Parameters
        ----------
        X : List[List[Any]]
            The data to predict on.
        mapping : Optional[dict], optional
            A mapping dictionary to map labels to their original values, by default None.

        Returns
        -------
        dict
            A dictionary containing the predicted labels and probabilities.
        """
        model = self.classifier_list[0]
        data_X = flatten(data_samples["X"])
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X=data_X)
            label = prob.argmax(axis=1)
            prob = reform_idx(prob, data_samples["X"])
        else:
            prob = None
            label = model.predict(X=data_X)

        if mapping is not None:
            label = [mapping[y] for y in label]

        label = reform_idx(label, data_samples["X"])

        return {"label": label, "prob": prob}

    def valid(self, X: List[List[Any]], Y: List[Any]) -> float:
        """
        Calculate the accuracy for the given data.

        Parameters
        ----------
        X : List[List[Any]]
            The data to calculate the accuracy on.
        Y : List[Any]
            The true labels for the given data.

        Returns
        -------
        float
            The accuracy score for the given data.
        """
        data_X = flatten(X)
        data_Y = flatten(Y)
        score = self.classifier_list[0].score(X=data_X, y=data_Y)
        return score

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
        data_X = flatten(data_samples["X"])
        data_y = flatten(data_samples["abduced_idx"])
        return self.classifier_list[0].fit(X=data_X, y=data_y)

    def _model_operation(self, operation: str, *args, **kwargs):
        model = self.classifier_list[0]
        if hasattr(model, operation):
            method = getattr(model, operation)
            method(*args, **kwargs)
        else:
            try:
                if not f"{operation}_path" in kwargs.keys():
                    raise ValueError(f"'{operation}_path' should not be None")
                if operation == "save":
                    with open(kwargs["save_path"], 'wb') as file:
                        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
                elif operation == "load":
                    with open(kwargs["load_path"], 'rb') as file:
                        self.classifier_list[0] = pickle.load(file)
            except:
                raise NotImplementedError(
                    f"{type(model).__name__} object doesn't have the {operation} method"
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
