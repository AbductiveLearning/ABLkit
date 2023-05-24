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
from typing import List, Any


def get_part_data(X, i):
    return list(map(lambda x: x[i], X))


def merge_data(X):
    ret_mark = list(map(lambda x: len(x), X))
    ret_X = list(chain(*X))
    return ret_X, ret_mark


def reshape_data(Y, marks):
    begin_mark = 0
    ret_Y = []
    for mark in marks:
        end_mark = begin_mark + mark
        ret_Y.append(Y[begin_mark:end_mark])
        begin_mark = end_mark
    return ret_Y


class ABLModel:
    """
    Serialize data and provide a unified interface for different machine learning models.

    Parameters
    ----------
    base_model : Machine Learning Model
        The base model to use for training and prediction.
    pseudo_label_list : List[Any]
        A list of pseudo labels to use for training.

    Attributes
    ----------
    cls_list : List[Any]
        A list of classifiers.
    pseudo_label_list : List[Any]
        A list of pseudo labels to use for training.
    mapping : dict
        A dictionary mapping pseudo labels to integers.
    remapping : dict
        A dictionary mapping integers to pseudo labels.

    Methods
    -------
    predict(X: List[List[Any]]) -> dict
        Predict the class labels and probabilities for the given data.
    valid(X: List[List[Any]], Y: List[Any]) -> float
        Calculate the accuracy score for the given data.
    train(X: List[List[Any]], Y: List[Any])
        Train the model on the given data.
    """
    def __init__(self, base_model, pseudo_label_list: List[Any]) -> None:
        self.cls_list = []
        self.cls_list.append(base_model)

        self.pseudo_label_list = pseudo_label_list
        self.mapping = dict(zip(pseudo_label_list, list(range(len(pseudo_label_list)))))
        self.remapping = dict(
            zip(list(range(len(pseudo_label_list))), pseudo_label_list)
        )

    def predict(self, X: List[List[Any]]) -> dict:
        """
        Predict the class labels and probabilities for the given data.

        Parameters
        ----------
        X : List[List[Any]]
            The data to predict on.

        Returns
        -------
        dict
            A dictionary containing the predicted class labels and probabilities.
        """
        data_X, marks = merge_data(X)
        prob = self.cls_list[0].predict_proba(X=data_X)
        _cls = prob.argmax(axis=1)
        cls = list(map(lambda x: self.remapping[x], _cls))

        prob = reshape_data(prob, marks)
        cls = reshape_data(cls, marks)

        return {"cls": cls, "prob": prob}

    def valid(self, X: List[List[Any]], Y: List[Any]) -> float:
        """
        Calculate the accuracy for the given data.

        Parameters
        ----------
        X : List[List[Any]]
            The data to calculate the accuracy on.
        Y : List[Any]
            The true class labels for the given data.

        Returns
        -------
        float
            The accuracy score for the given data.
        """
        data_X, _ = merge_data(X)
        _data_Y, _ = merge_data(Y)
        data_Y = list(map(lambda y: self.mapping[y], _data_Y))
        score = self.cls_list[0].score(X=data_X, y=data_Y)
        return score

    def train(self, X: List[List[Any]], Y: List[Any]):
        """
        Train the model on the given data.

        Parameters
        ----------
        X : List[List[Any]]
            The data to train on.
        Y : List[Any]
            The true class labels for the given data.
        """
        data_X, _ = merge_data(X)
        _data_Y, _ = merge_data(Y)
        data_Y = list(map(lambda y: self.mapping[y], _data_Y))
        self.cls_list[0].fit(X=data_X, y=data_Y)
