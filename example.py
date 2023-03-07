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

from utils.plog import logger, INFO
from utils.utils import reduce_dimension
import torch.nn as nn
import torch

from models.nn import LeNet5, SymbolNet
from models.basic_model import BasicModel, BasicDataset
from models.wabl_models import DecisionTree, WABLBasicModel
from sklearn.neighbors import KNeighborsClassifier

from multiprocessing import Pool
from abducer.abducer_base import AbducerBase
from abducer.kb import add_KB, HWF_KB, HED_prolog_KB
from datasets.mnist_add.get_mnist_add import get_mnist_add
from datasets.hwf.get_hwf import get_hwf
from datasets.hed.get_hed import get_hed, split_equation
import framework_hed
import framework_hed_knn


def run_test():

    # kb = add_KB(True)
    # kb = HWF_KB(True)
    # abducer = AbducerBase(kb)

    kb = HED_prolog_KB()
    abducer = AbducerBase(kb, zoopt=True, multiple_predictions=True)

    recorder = logger()

    total_train_data = get_hed(train=True)
    train_data, val_data = split_equation(total_train_data, 3, 1)
    test_data = get_hed(train=False)

    # ======================== non-NN model ========================== #
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

    # ========================== NN model ============================ #
    # # cls = LeNet5(num_classes=len(kb.pseudo_label_list), image_size=(train_data[0][0][0].shape[1:]))
    # cls = SymbolNet(num_classes=len(kb.pseudo_label_list))
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # framework_hed.hed_pretrain(kb, cls, recorder)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.RMSprop(cls.parameters(), lr=0.001, weight_decay=1e-6)
    # # optimizer = torch.optim.Adam(cls.parameters(), lr=0.00001, betas=(0.9, 0.99))

    # base_model = BasicModel(cls, criterion, optimizer, device, save_interval=1, save_dir=recorder.save_dir, batch_size=32, num_epochs=10, recorder=recorder)
    # model = WABLBasicModel(base_model, kb.pseudo_label_list)

    # # train_X, train_Z, train_Y = get_mnist_add(train = True, get_pseudo_label = True)
    # # test_X, test_Z, test_Y = get_mnist_add(train = False, get_pseudo_label = True)

    # # train_data = get_hwf(train = True, get_pseudo_label = True)
    # # test_data = get_hwf(train = False, get_pseudo_label = True)

    # model, mapping = framework_hed.train_with_rule(model, abducer, train_data, val_data, select_num=10, min_len=5, max_len=8)
    # framework_hed.hed_test(model, abducer, mapping, train_data, test_data, min_len=5, max_len=8)
    # ============================ End =============================== #

    recorder.dump()
    return True


if __name__ == "__main__":
    run_test()
