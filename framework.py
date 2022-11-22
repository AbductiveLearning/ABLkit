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

def block_sample(X, Z, Y, sample_num, epoch_idx):
    part_num = (len(X) // sample_num)
    if part_num == 0:
        part_num = 1
    seg_idx = epoch_idx % part_num 
    INFO("seg_idx:", seg_idx, ", part num:", part_num, ", data num:", len(X))
    X = X[sample_num * seg_idx: sample_num * (seg_idx + 1)]
    Z = Z[sample_num * seg_idx: sample_num * (seg_idx + 1)]
    Y = Y[sample_num * seg_idx: sample_num * (seg_idx + 1)]

    return X, Z, Y

def get_taglist(self, Z):
    tmp = [[str(x) for x in label] for label in Z]
    tmp = sorted(list(set(tmp)))
    return tmp

def get_abl_acc(pred_Z, Y, logic_forward):
    abl_acc = 0
    for pred_z, y in zip(pred_Z, Y):
        if(logic_forward(pred_z) == y):
            abl_acc += 1      
    return abl_acc / len(Y)

def get_char_acc(Z, pred_Z):
    char_acc = 0
    char_num = 0
    for pred_z, z in zip(pred_Z, Z):
        char_num += len(z)
        for zidx in range(len(z)):
            if(pred_z[zidx] == z[zidx]):
                char_acc += 1
    return char_acc / char_num
    

def filter_data(X, abduced_Z):
    finetune_Z = []
    finetune_X = []
    for abduced_x, abduced_z in zip(X, abduced_Z):
        if abduced_z is not []:
            finetune_X.append(abduced_x)
            finetune_Z.append(abduced_z)
    return finetune_X, finetune_Z

def pretrain(model, X, Z):
    pass

def train(model, abducer, train_data, test_data, epochs = 5, sample_num = -1, verbose = -1):
    X, Z, Y = train_data
    test_X, test_Z, test_Y = test_data
    
    # Set default parameters
    if sample_num == -1:
        sample_num = len(X)

    if verbose < 1:
        verbose = epochs
    
    char_acc_flag = 1
    if Z == None:
        char_acc_flag = 0
        Z = [None] * len(X)

    predict_func = clocker(model.predict)
    train_func = clocker(model.train)
    abduce_func = clocker(abducer.batch_abduce)
    
    # Abductive learning train process
    for epoch_idx in range(epochs):
        X, Z, Y = block_sample(X, Z, Y, sample_num, epoch_idx)
        preds_res = predict_func(X)
        abduced_Z = abduce_func(preds_res, Y)

        abl_acc = get_abl_acc(preds_res['cls'], Y, abducer.kb.logic_forward)
        if(char_acc_flag):
            ori_char_acc = get_char_acc(preds_res['cls'], Z)
            abd_char_acc = get_char_acc(preds_res['cls'], abduced_Z)
            INFO('epoch_idx:', epoch_idx, '  abl_acc:', abl_acc, '  ori_char_acc:', ori_char_acc, '  abd_char_acc:', abd_char_acc)
        else:
            INFO('epoch_idx:', epoch_idx, '  abl_acc:', abl_acc)
        
        finetune_X, finetune_Z = filter_data(X, abduced_Z)
        if len(finetune_X) > 0:
            # model.valid(finetune_X, finetune_Z)
            train_func(finetune_X, finetune_Z)
        else:
            INFO("lack of data, all abduced failed", len(finetune_X))
            
    return abl_acc

if __name__ == "__main__":
    pass
