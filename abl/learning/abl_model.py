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
from itertools import chain
from typing import List, Any, Optional


def get_part_data(X, i):
    return list(map(lambda x: x[i], X))


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
    predict(X: List[List[Any]], mapping: Optional[dict]) -> dict
        Predict the labels and probabilities for the given data.
    valid(X: List[List[Any]], Y: List[Any]) -> float
        Calculate the accuracy score for the given data.
    train(X: List[List[Any]], Y: List[Any])
        Train the model on the given data.
    """

    def __init__(self, base_model) -> None:
        self.classifier_list = []
        self.classifier_list.append(base_model)

    def predict(self, X: List[List[Any]], mapping: Optional[dict] = None) -> dict:
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
        data_X, marks = self.merge_data(X)
        prob = self.classifier_list[0].predict_proba(X=data_X)
        label = prob.argmax(axis=1)
        if mapping is not None:
            label = [mapping[x] for x in label]

        prob = self.reshape_data(prob, marks)
        label = self.reshape_data(label, marks)

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
        data_X, _ = self.merge_data(X)
        data_Y, _ = self.merge_data(Y)
        score = self.classifier_list[0].score(X=data_X, y=data_Y)
        return score

    def train(self, X: List[List[Any]], Y: List[Any]) -> float:
        """
        Train the model on the given data.

        Parameters
        ----------
        X : List[List[Any]]
            The data to train on.
        Y : List[Any]
            The true labels for the given data.
        """
        data_X, _ = self.merge_data(X)
        data_Y, _ = self.merge_data(Y)
        return self.classifier_list[0].fit(X=data_X, y=data_Y)

    @staticmethod
    def merge_data(X):
        ret_mark = list(map(lambda x: len(x), X))
        ret_X = list(chain(*X))
        return ret_X, ret_mark

    @staticmethod
    def reshape_data(Y, marks):
        begin_mark = 0
        ret_Y = []
        for mark in marks:
            end_mark = begin_mark + mark
            ret_Y.append(list(Y[begin_mark:end_mark]))
            begin_mark = end_mark
        return ret_Y
