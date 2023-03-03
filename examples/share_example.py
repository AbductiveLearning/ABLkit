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

from multiprocessing import Pool
import os
from datasets.data_generator import generate_data_via_codes, code_generator
from collections import defaultdict
from abducer.abducer_base import AbducerBase
from abducer.kb import ClsKB, RegKB

def run_test(params):
    code_len, times, code_num, share, model_type, need_prob, letter_num = params

    if share:
        result_dir = "share_result"
    else:
        result_dir = "non_share_result"

    recoder_file_path = f"{result_dir}/random_{times}_{code_len}_{code_num}_{model_type}_{need_prob}.pk"#

    words = code_generator(code_len, code_num, letter_num)
    kb = ClsKB(words)
    abducer = AbducerBase(kb)

    label_lists = [[] for _ in range(code_len)]
    for widx, word in enumerate(words):
        for cidx, c in enumerate(word):
            label_lists[cidx].append(c)

    if share:
        label_lists = [sum(label_lists, [])]

    recoder = logger()
    recoder.set_savefile("test.log")
    for idx, err in enumerate(range(0, 41)):
        print("Start expriment", idx)
        start = time.process_time()
        err = err / 40.
        if 1 - err < (1. / letter_num):
            break
        if model_type == "KNN":
            model = KNN(code_len, label_lists = label_lists, share=share)
        elif model_type == "DT":
            model = DecisionTree(code_len, label_lists = label_lists, share=share)

        pre_X, pre_Y = generate_data_via_codes(words, err, letter_num)
        X, Y = generate_data_via_codes(words, 0, letter_num)

        str_words = ["".join(str(c) for c in word) for word in words]

        recoder.print(str_words)

        model.train(pre_X, pre_Y)
        abl_epoch = 30
        res = framework.train(model, abducer, X, Y, sample_num = 10000, verbose = 1)
        print("Initial data accuracy:", 1 - err)
        print("Abd word accuracy:    ", res["accuracy_word"] * 1.0 / res["total_word"])
        print("Abd char accuracy:    ", res["accuracy_abd_char"] * 1.0 / res["total_abd_char"])
        print("Ori char accuracy:    ", res["accuracy_ori_char"] * 1.0 / res["total_ori_char"])
        print("End expriment", idx)
        print()

    recoder.dump(open(recoder_file_path, "wb"))
    return True

if __name__ == "__main__":
    os.system("mkdir share_result")
    os.system("mkdir non_share_result")
    
    for times in range(5):
        for code_num in [32, 64, 128]:
            params = [11, times, code_num, True, "KNN", True, 2]
            run_test(params)

            params = [11, times, code_num, True, "KNN", False, 2]
            run_test(params)

    #params = [11, 0, 32, True, "DT", True, 2]
    #run_test(params)

