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
import numpy as np
import time
import framework
import utils.plog as plog
import torch.nn as nn
import torch

from models.lenet5 import LeNet5
from models.basic_model import BasicModel
from models.wabl_models import MyModel

from multiprocessing import Pool
import os
from abducer.abducer_base import AbducerBase
from abducer.kb import add_KB, hwf_KB
from datasets.mnist_add.get_mnist_add import get_mnist_add
from datasets.hwf.get_hwf import get_hwf

class Params:
    imgH = 28
    imgW = 28
    keep_ratio = True
    saveInterval = 10
    batchSize = 16
    workers = 16
    n_epoch = 10
    stop_loss = None

def run_test():
    
    result_dir = 'results'

    recorder_file_path = f"{result_dir}/1116.pk"#

    kb = add_KB()
    # kb = hwf_KB()
    abducer = AbducerBase(kb, 2)

    recorder = logger()
    recorder.set_savefile("test.log")

    train_X, train_Z, train_Y = get_mnist_add(train = True, get_pseudo_label = True)
    test_X, test_Z, test_Y = get_mnist_add(train = False, get_pseudo_label = True)

    recorder = plog.ResultRecorder()
    cls = LeNet5(num_classes=len(kb.pseudo_label_list), image_size=(train_X[0][0].shape[1:]))
    
    criterion = nn.CrossEntropyLoss(size_average=True)
    optimizer = torch.optim.Adam(cls.parameters(), lr=0.001, betas=(0.9, 0.99))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    base_model = BasicModel(cls, criterion, optimizer, device, Params(), recorder=recorder)
    model = MyModel(base_model, kb.pseudo_label_list)

    res = framework.train(model, abducer, train_X, train_Z, train_Y, sample_num = 10000, verbose = 1)
    print(res)
    

    recorder.dump(open(recorder_file_path, "wb"))
    return True

if __name__ == "__main__":
    os.system("mkdir results")
    
    run_test()

