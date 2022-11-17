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
from models.wabl_models import DecisionTree, KNN
import pickle as pk
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
from datasets.data_generator import generate_data_via_codes, code_generator
from collections import defaultdict
from abducer.abducer_base import AbducerBase
from abducer.kb import add_KB, hwf_KB
from datasets.mnist_add.get_mnist_add import get_mnist_add
from datasets.hwf.get_hwf import get_hwf

class Params:
    imgH = 45
    imgW = 45
    keep_ratio = True
    saveInterval = 10
    batchSize = 16
    workers = 16
    n_epoch = 10
    stop_loss = None

def run_test():
    
    result_dir = 'results'

    recorder_file_path = f"{result_dir}/1116.pk"#

    # words = code_generator(code_len, code_num, letter_num)
    kb = add_KB()
    abducer = AbducerBase(kb)

    recorder = logger()
    recorder.set_savefile("test.log")


    train_X, train_Y, test_X, test_Y = get_mnist_add()
    # train_X, train_Y, test_X, test_Y = get_hwf()


    recorder = plog.ResultRecorder()
    cls = LeNet5()
    
    criterion = nn.CrossEntropyLoss(size_average=True)
    optimizer = torch.optim.Adam(cls.parameters(), lr=0.001, betas=(0.9, 0.99))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sign_list = list(range(10))
    base_model = BasicModel(cls, criterion, optimizer, device, Params(), sign_list, recorder=recorder)
    model = MyModel(base_model)

    res = framework.train(model, abducer, train_X, train_Y, logic_forward = kb.logic_forward, sample_num = 10000, verbose = 1)
    print(res)
    

    recorder.dump(open(recorder_file_path, "wb"))
    return True

if __name__ == "__main__":
    os.system("mkdir results")
    
    run_test()

