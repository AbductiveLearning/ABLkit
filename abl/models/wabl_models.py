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


class WABLBasicModel:
    def __init__(self, base_model, pseudo_label_list):
        self.cls_list = []
        self.cls_list.append(base_model)

        self.pseudo_label_list = pseudo_label_list
        self.mapping = dict(zip(pseudo_label_list, list(range(len(pseudo_label_list)))))
        self.remapping = dict(
            zip(list(range(len(pseudo_label_list))), pseudo_label_list)
        )

    def predict(self, X):
        data_X, marks = merge_data(X)
        prob = self.cls_list[0].predict_proba(X=data_X)
        _cls = prob.argmax(axis=1)
        cls = list(map(lambda x: self.remapping[x], _cls))

        prob = reshape_data(prob, marks)
        cls = reshape_data(cls, marks)

        return {"cls": cls, "prob": prob}

    def valid(self, X, Y):
        data_X, _ = merge_data(X)
        _data_Y, _ = merge_data(Y)
        data_Y = list(map(lambda y: self.mapping[y], _data_Y))
        score = self.cls_list[0].score(X=data_X, y=data_Y)
        return score, [score]

    def train(self, X, Y):
        # self.label_lists = []
        data_X, _ = merge_data(X)
        _data_Y, _ = merge_data(Y)
        data_Y = list(map(lambda y: self.mapping[y], _data_Y))
        self.cls_list[0].fit(X=data_X, y=data_Y)
