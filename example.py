# coding: utf-8
#================================================================#
#   Copyright (C) 2021 Freecss All rights reserved.
#   
#   File Name     ：share_example.py
#   Author        ：freecss
#   Email         ：karlfreecss@gmail.com
#   Created Date  ：2021/06/07
#   Description   ：
#
#================================================================#

from utils.plog import logger
import framework
import torch.nn as nn
import torch

from models.lenet5 import LeNet5, SymbolNet
from models.basic_model import BasicModel
from models.wabl_models import WABLBasicModel

from multiprocessing import Pool
import os
from abducer.abducer_base import AbducerBase
from abducer.kb import add_KB, hwf_KB
from datasets.mnist_add.get_mnist_add import get_mnist_add
from datasets.hwf.get_hwf import get_hwf

def run_test():

    # kb = add_KB(True)
    kb = hwf_KB(True)
    abducer = AbducerBase(kb)

    recorder = logger()

    # train_X, train_Z, train_Y = get_mnist_add(train = True, get_pseudo_label = True)
    # test_X, test_Z, test_Y = get_mnist_add(train = False, get_pseudo_label = True)
    
    train_data = get_hwf(train = True, get_pseudo_label = True)
    test_data = get_hwf(train = False, get_pseudo_label = True)

    # cls = LeNet5(num_classes=len(kb.pseudo_label_list), image_size=(train_data[0][0][0].shape[1:]))
    cls = SymbolNet(num_classes=len(kb.pseudo_label_list))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cls.parameters(), lr=0.001, betas=(0.9, 0.99))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    base_model = BasicModel(cls, criterion, optimizer, device, save_interval=1, save_dir=recorder.save_dir, num_epochs=1, recorder=recorder)
    model = WABLBasicModel(base_model, kb.pseudo_label_list)

    res = framework.train(model, abducer, train_data, test_data, sample_num = 10000, verbose = 1)
    recorder.print(res)
    
    recorder.dump()
    return True

if __name__ == "__main__":
    run_test()
