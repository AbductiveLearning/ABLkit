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
from utils.utils import copy_state_dict
import torch.nn as nn
import torch

from models.nn import LeNet5, SymbolNet, SymbolNetAutoencoder
from models.basic_model import BasicModel, BasicDataset
from models.wabl_models import DecisionTree, WABLBasicModel

from multiprocessing import Pool
import os
from abducer.abducer_base import AbducerBase
from abducer.kb import add_KB, HWF_KB, HED_prolog_KB
from datasets.mnist_add.get_mnist_add import get_mnist_add
from datasets.hwf.get_hwf import get_hwf
from datasets.hed.get_hed import get_hed, split_equation, get_pretrain_data
import framework_hed


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
    
    # cls = LeNet5(num_classes=len(kb.pseudo_label_list), image_size=(train_data[0][0][0].shape[1:]))
    cls_autoencoder = SymbolNetAutoencoder(num_classes=len(kb.pseudo_label_list))
    cls = SymbolNet(num_classes=len(kb.pseudo_label_list))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists("./weights/pretrain_weights.pth"):
        INFO("Pretrain Start")
        pretrain_data_X, pretrain_data_Y = get_pretrain_data(['0', '1', '10', '11'])
        pretrain_data = BasicDataset(pretrain_data_X, pretrain_data_Y)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.RMSprop(cls_autoencoder.parameters(), lr=0.001, alpha=0.9, weight_decay=1e-6)

        pretrain_model = BasicModel(cls_autoencoder, criterion, optimizer, device, save_interval=1, save_dir=recorder.save_dir, num_epochs=10, recorder=recorder)
        framework_hed.pretrain(pretrain_model, pretrain_data)
        torch.save(cls_autoencoder.base_model.state_dict(), "./weights/pretrain_weights.pth")
        cls.load_state_dict(cls_autoencoder.base_model.state_dict())
    
    else:
        cls.load_state_dict(torch.load("./weights/pretrain_weights.pth"))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(cls.parameters(), lr=0.001, weight_decay=1e-6)
    # optimizer = torch.optim.Adam(cls.parameters(), lr=0.00001, betas=(0.9, 0.99))
    

    base_model = BasicModel(cls, criterion, optimizer, device, save_interval=1, save_dir=recorder.save_dir, batch_size=32, num_epochs=10, recorder=recorder)
    model = WABLBasicModel(base_model, kb.pseudo_label_list)
    
    # train_X, train_Z, train_Y = get_mnist_add(train = True, get_pseudo_label = True)
    # test_X, test_Z, test_Y = get_mnist_add(train = False, get_pseudo_label = True)

    # train_data = get_hwf(train = True, get_pseudo_label = True)
    # test_data = get_hwf(train = False, get_pseudo_label = True)

    framework_hed.train_with_rule(model, abducer, train_data, val_data, select_num=10, verbose=1)
    # recorder.print(res)

    recorder.dump()
    return True


if __name__ == "__main__":
    run_test()
