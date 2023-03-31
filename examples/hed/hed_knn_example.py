# coding: utf-8
# ================================================================#
#   Copyright (C) 2021 Freecss All rights reserved.
#
#   File Name     ：share_example.py
#   Author        ：freecss
#   Email         ：karlfreecss@gmail.com
#   Created Date  ：2021/06/07
#   Description   ：
#
# ================================================================#

import sys
sys.path.append("../")

from abl.utils.plog import logger, INFO
from abl.utils.utils import reduce_dimension
import torch.nn as nn
import torch

from abl.models.nn import LeNet5, SymbolNet
from abl.models.basic_model import BasicModel, BasicDataset
from abl.models.wabl_models import DecisionTree, WABLBasicModel
from sklearn.neighbors import KNeighborsClassifier

from abl.abducer.abducer_base import AbducerBase
from abl.abducer.kb import add_KB, HWF_KB, prolog_KB
from datasets.mnist_add.get_mnist_add import get_mnist_add
from datasets.hwf.get_hwf import get_hwf
from datasets.hed.get_hed import get_hed, split_equation
from abl import framework_hed_knn


def run_test():

    # kb = add_KB(True)
    # kb = HWF_KB(True)
    # abducer = AbducerBase(kb)

    kb = prolog_KB(pseudo_label_list=[1, 0, '+', '='], pl_file='../examples/datasets/hed/learn_add.pl')
    abducer = AbducerBase(kb, zoopt=True, multiple_predictions=True)

    recorder = logger()

    total_train_data = get_hed(train=True)
    train_data, val_data = split_equation(total_train_data, 3, 1)
    test_data = get_hed(train=False)

    # ========================= KNN model ============================ #
    reduce_dimension(train_data)
    reduce_dimension(val_data)
    reduce_dimension(test_data)
    base_model = KNeighborsClassifier(n_neighbors=3)
    pretrain_data_X, pretrain_data_Y = framework_hed_knn.hed_pretrain(base_model)
    model = WABLBasicModel(base_model, kb.pseudo_label_list)
    model, mapping = framework_hed_knn.train_with_rule(
        model, abducer, train_data, val_data, (pretrain_data_X, pretrain_data_Y), select_num=10, min_len=5, max_len=8
    )
    framework_hed_knn.hed_test(
        model, abducer, mapping, train_data, test_data, min_len=5, max_len=8
    )
    # ============================ End =============================== #

    recorder.dump()
    return True


if __name__ == "__main__":
    run_test()
