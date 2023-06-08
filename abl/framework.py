# coding: utf-8
# ================================================================#
#   Copyright (C) 2021 Freecss All rights reserved.
#
#   File Name     ：framework.py
#   Author        ：freecss
#   Email         ：karlfreecss@gmail.com
#   Created Date  ：2021/06/07
#   Description   ：
#
# ================================================================#

from .utils.plog import INFO, clocker
from .utils.utils import block_sample, float_parameter


def result_statistics(pred_Z, Z, Y, logic_forward, char_acc_flag):
    result = {}
    if char_acc_flag:
        char_acc_num = 0
        char_num = 0
        for pred_z, z in zip(pred_Z, Z):
            char_num += len(z)
            for zidx in range(len(z)):
                if pred_z[zidx] == z[zidx]:
                    char_acc_num += 1
        char_acc = char_acc_num / char_num
        result["Character level accuracy"] = char_acc

    abl_acc_num = 0
    for pred_z, y in zip(pred_Z, Y):
        if logic_forward(pred_z) == y:
            abl_acc_num += 1
    abl_acc = abl_acc_num / len(Y)
    result["ABL accuracy"] = abl_acc

    return result


def filter_data(X, abduced_Z):
    finetune_Z = []
    finetune_X = []
    for x, abduced_z in zip(X, abduced_Z):
        if len(abduced_z) > 0:
            finetune_X.append(x)
            finetune_Z.append(abduced_z)
    return finetune_X, finetune_Z


def train(model, abducer, train_data, epochs=50, sample=-1, verbose=-1):
    train_X, train_Z, train_Y = train_data

    # Set default parameters
    sample_num = float_parameter(sample, len(train_X))
    part_num = (len(train_X) - 1) // sample_num + 1

    if verbose < 1:
        verbose = epochs

    char_acc_flag = 1
    if train_Z == None:
        char_acc_flag = 0
        train_Z = [None] * len(train_X)

    predict_func = clocker(model.predict)
    train_func = clocker(model.train)
    abduce_func = clocker(abducer.batch_abduce)
    
    for epoch in range(epochs):
        for seg_idx in range(part_num):
            X, Z, Y = block_sample(train_X, train_Z, train_Y, sample_num, seg_idx)
            INFO("epoch:", epoch + 1, ", seg_idx:", seg_idx + 1, "/", part_num, ", data num:", len(X))
            
            preds_res = predict_func(X)
            abduced_Z = abduce_func(preds_res, Y)

            ## TODO: change verbose
            if ((seg_idx + 1) % verbose == 0) or (seg_idx == epochs - 1):
                pseudo_label = [[abducer.mapping[label] for label in formula] for formula in preds_res['label']]
                res = result_statistics(pseudo_label, Z, Y, abducer.kb.logic_forward, char_acc_flag)
                INFO("seg: ", seg_idx + 1, " ", res)

            finetune_X, finetune_Z = filter_data(X, abduced_Z)
            finetune_Z = [[abducer.remapping[symbol] for symbol in formula] for formula in finetune_Z]
            if len(finetune_X) > 0:
                # model.valid(finetune_X, finetune_Z)
                train_func(finetune_X, finetune_Z)
            else:
                INFO("lack of data, all abduced failed", len(finetune_X))

    return model

## TODO: test
def test(model, abducer, test_data):
    test_X, test_Z, test_Y = test_data
    predict_func = clocker(model.predict)
    preds_res = predict_func(test_X)
    
    char_acc_flag = 1
    if test_Z == None:
        char_acc_flag = 0
        test_Z = [None] * len(test_X)
        
    res = result_statistics(preds_res["cls"], test_Z, test_Y, abducer.kb.logic_forward, char_acc_flag)
    INFO(res)

if __name__ == "__main__":
    pass
