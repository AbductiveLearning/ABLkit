# coding: utf-8
#================================================================#
#   Copyright (C) 2021 Freecss All rights reserved.
#   
#   File Name     ：framework.py
#   Author        ：freecss
#   Email         ：karlfreecss@gmail.com
#   Created Date  ：2021/06/07
#   Description   ：
#
#================================================================#

import pickle as pk

import numpy as np

from utils.plog import INFO, DEBUG, clocker

@clocker
def block_sample(X_bak, Y_bak, C_bak, sample_num, epoch_idx):
    part_num = (len(X_bak) // sample_num)
    if part_num == 0:
        part_num = 1
    seg_idx = epoch_idx % part_num 
    INFO("seg_idx:", seg_idx, ", part num:", part_num, ", data num:", len(X_bak))
    X = X_bak[sample_num * seg_idx: sample_num * (seg_idx + 1)]
    Y = Y_bak[sample_num * seg_idx: sample_num * (seg_idx + 1)]
    C = C_bak[sample_num * seg_idx: sample_num * (seg_idx + 1)]

    return X, Y, C

def get_taglist(self, Y):
    tmp = [[str(x) for x in label] for label in Y]
    tmp = sorted(list(set(tmp)))
    return tmp

@clocker
def result_statistics(pseudo_Y, Y, abduced_Y):

    abd_err_num = 0
    abd_char_num = 0
    abd_char_acc = 0
    abd_failed = 0
    word_err_num = 0
    
    ori_char_num = 0
    ori_char_acc = 0

    for tidx, (pseudo_y, y, abduced_y) in enumerate(zip(pseudo_Y, Y, abduced_Y)):
        pseudo_y = pseudo_y
        if sum(abduced_y != y) != 0:
            abd_err_num += 1
        if abduced_y is not None:
            abd_char_num += len(y)
            abd_char_acc += sum(abduced_y == y)
        else:
            abd_failed += 1

        ori_char_num += len(pseudo_y)
        ori_char_acc += sum(pseudo_y == y)
        
        if abduced_y is not None and sum(y != pseudo_y) == 0 and sum(pseudo_y != abduced_y) > 0:
            INFO(pseudo_y, y, abduced_y)
            pk.dump((pseudo_y, y, abduced_y), open("bug.pk", "wb"))

        if sum(pseudo_y != y) != 0:
            word_err_num += 1

    INFO("")
    INFO("Abd word level accuracy:", 1 - word_err_num / len(pseudo_Y))
    INFO("Abd char level accuracy:", abd_char_acc / abd_char_num)
    INFO("Ori char level accuracy:", ori_char_acc / ori_char_num)
    INFO("")

    result = {"total_word" : len(pseudo_Y), "accuracy_word" : len(pseudo_Y) - word_err_num,
              "total_abd_char": abd_char_num, "accuracy_abd_char" : abd_char_acc,
              "total_ori_char": ori_char_num, "accuracy_ori_char" : ori_char_acc,
              "total_abd_failed": abd_failed}

    return result

@clocker
def filter_data(X, abduced_Y):
    finetune_Y = []
    finetune_X = []
    for abduced_x, abduced_y in zip(X, abduced_Y):
        if abduced_y is not None:
            finetune_X.append(abduced_x)
            finetune_Y.append(abduced_y)
    return finetune_X, finetune_Y

@clocker
def is_all_sublabel_exist(labels, std_label_list):
    if not labels:
        return False
    
    labels = np.array(labels).T
    for idx, (std_label, label) in enumerate(zip(std_label_list, labels)):
        std_num = len(set(std_label))
        sublabel_num = len(set(label))
        if std_num != sublabel_num:
            INFO(f"sublabel {idx} should have {std_num} class, but data only have {sublabel_num} class", screen=True)
            return False
    return True

def pretrain(model, X, Y):
    pass

def train(model, abducer, X, Y, C = None, epochs = 10, sample_num = -1, verbose = -1, check_sublabel = True):
    # Set default parameters
    if sample_num == -1:
        sample_num = len(X)

    if verbose < 1:
        verbose = epochs

    if C is None:
        C = [None] * len(X)

    # Set function running time recorder
    valid_func = clocker(model.valid)
    predict_func = clocker(model.predict)
    train_func = clocker(model.train)

    abduce_func = clocker(abducer.batch_abduce)

    X_bak = X
    Y_bak = Y
    C_bak = C

    # Abductive learning train process
    res = {}
    for epoch_idx in range(epochs):
        X, Y, C = block_sample(X_bak, Y_bak, C_bak, sample_num, epoch_idx)
        preds_res = predict_func(X)
        abduced_Y = abduce_func(preds_res, C)
        finetune_X, finetune_Y = filter_data(X, abduced_Y)
        score, score_list = valid_func(X, Y)
        if ((epoch_idx + 1) % verbose == 0) or (epoch_idx == epochs - 1):
            res = result_statistics(preds_res["cls"], Y, abduced_Y)
            INFO(res)

        if check_sublabel and (not is_all_sublabel_exist(finetune_Y, model.label_lists)):
            INFO("There is some sub label missing", len(finetune_Y))
            break

        if len(finetune_X) > 0:
            train_func(finetune_X, finetune_Y)#, n_epoch = 10)
        else:
            INFO("lack of data, all abduced failed", len(finetune_X))
    return res
    #return ret

if __name__ == "__main__":
    pass
