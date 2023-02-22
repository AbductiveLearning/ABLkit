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

import pickle as pk
import torch
import torch.nn as nn
import numpy as np
import os

from utils.plog import INFO, DEBUG, clocker
from utils.utils import flatten, reform_idx, block_sample, gen_mappings, mapping_res, remapping_res

from models.nn import MLP, SymbolNetAutoencoder
from models.basic_model import BasicModel, BasicDataset
from datasets.hed.get_hed import get_pretrain_data

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
    for abduced_x, abduced_z in zip(X, abduced_Z):
        if abduced_z is not []:
            finetune_X.append(abduced_x)
            finetune_Z.append(abduced_z)
    return finetune_X, finetune_Z


    
    
def train(model, abducer, train_data, test_data, epochs=50, sample_num=-1, verbose=-1):
    train_X, train_Z, train_Y = train_data
    test_X, test_Z, test_Y = test_data

    # Set default parameters
    if sample_num == -1:
        sample_num = len(train_X)

    if verbose < 1:
        verbose = epochs

    char_acc_flag = 1
    if train_Z == None:
        char_acc_flag = 0
        train_Z = [None] * len(train_X)

    predict_func = clocker(model.predict)
    train_func = clocker(model.train)
    abduce_func = clocker(abducer.batch_abduce)

    for epoch_idx in range(epochs):
        X, Z, Y = block_sample(train_X, train_Z, train_Y, sample_num, epoch_idx)
        preds_res = predict_func(X)
        # input()
        abduced_Z = abduce_func(preds_res, Y)

        if ((epoch_idx + 1) % verbose == 0) or (epoch_idx == epochs - 1):
            res = result_statistics(preds_res['cls'], Z, Y, abducer.kb.logic_forward, char_acc_flag)
            INFO('epoch: ', epoch_idx + 1, ' ', res)

        finetune_X, finetune_Z = filter_data(X, abduced_Z)
        if len(finetune_X) > 0:
            # model.valid(finetune_X, finetune_Z)
            train_func(finetune_X, finetune_Z)
        else:
            INFO("lack of data, all abduced failed", len(finetune_X))

    return res


def hed_pretrain(kb, cls, recorder):
    cls_autoencoder = SymbolNetAutoencoder(num_classes=len(kb.pseudo_label_list))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not os.path.exists("./weights/pretrain_weights.pth"):
        INFO("Pretrain Start")
        pretrain_data_X, pretrain_data_Y = get_pretrain_data(['0', '1', '10', '11'])
        pretrain_data = BasicDataset(pretrain_data_X, pretrain_data_Y)
        pretrain_data_loader = torch.utils.data.DataLoader(pretrain_data, batch_size=64, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.RMSprop(cls_autoencoder.parameters(), lr=0.001, alpha=0.9, weight_decay=1e-6)

        pretrain_model = BasicModel(cls_autoencoder, criterion, optimizer, device, save_interval=1, save_dir=recorder.save_dir, num_epochs=10, recorder=recorder)
        pretrain_model.fit(pretrain_data_loader)
        torch.save(cls_autoencoder.base_model.state_dict(), "./weights/pretrain_weights.pth")
        cls.load_state_dict(cls_autoencoder.base_model.state_dict())
    
    else:
        cls.load_state_dict(torch.load("./weights/pretrain_weights.pth"))


def get_char_acc(model, X, consistent_pred_res, mapping):
    original_pred_res = model.predict(X)['cls']
    pred_res = flatten(mapping_res(original_pred_res, mapping))
    INFO('Current model\'s output: ', pred_res)
    INFO('Abduced labels:         ', flatten(consistent_pred_res))
    assert len(pred_res) == len(flatten(consistent_pred_res))
    return sum([pred_res[idx] == flatten(consistent_pred_res)[idx] for idx in range(len(pred_res))]) / len(pred_res)


def abduce_and_train(model, abducer, mapping, train_X_true, select_num):
    select_idx = np.random.randint(len(train_X_true), size=select_num)
    X = []
    for idx in select_idx:
        X.append(train_X_true[idx])

    original_pred_res = model.predict(X)['cls']
    
    if mapping == None:
        mappings = gen_mappings(['+', '=', 0, 1],['+', '=', 0, 1])
    else:
        mappings = [mapping]
    
    consistent_idx = []
    consistent_pred_res = []
    
    for m in mappings:
        pred_res = mapping_res(original_pred_res, m)
        max_abduce_num = 20
        solution = abducer.zoopt_get_solution(pred_res, [1] * len(pred_res), max_abduce_num)
        all_address_flag = reform_idx(solution, pred_res)

        consistent_idx_tmp = []
        consistent_pred_res_tmp = []
        
        for idx in range(len(pred_res)):
            address_idx = [i for i, flag in enumerate(all_address_flag[idx]) if flag != 0]
            candidate = abducer.kb.address_by_idx([pred_res[idx]], 1, address_idx, True)
            if len(candidate) > 0:
                consistent_idx_tmp.append(idx)
                consistent_pred_res_tmp.append(candidate[0][0])
        
        if len(consistent_idx_tmp) > len(consistent_idx):
            consistent_idx = consistent_idx_tmp
            consistent_pred_res = consistent_pred_res_tmp
            if len(mappings) > 1:
                mapping = m
                
    if len(consistent_idx) == 0:
        return 0, 0, None
    
    if len(mappings) > 1:
        INFO('Final mapping is: ', mapping)
    
    INFO('Train pool size is:', len(flatten(consistent_pred_res)))
    INFO("Start to use abduced pseudo label to train model...")
    model.train([X[idx] for idx in consistent_idx], remapping_res(consistent_pred_res, mapping))

    consistent_acc = len(consistent_idx) / select_num
    char_acc = get_char_acc(model, [X[idx] for idx in consistent_idx], consistent_pred_res, mapping)
    INFO('consistent_acc is %s, char_acc is %s' % (consistent_acc, char_acc))
    return consistent_acc, char_acc, mapping


def output_rules(rules):
    all_rule_dict = {}
    for rule in rules:
        for r in rule:
            all_rule_dict[r] = 1 if r not in all_rule_dict else all_rule_dict[r] + 1
    rule_dict = {rule: cnt for rule, cnt in all_rule_dict.items()}# if cnt >= 5}
    return rule_dict

def get_rules_from_data(model, abducer, mapping, train_X_true, samples_per_rule, logic_output_dim):
    rules = []
    for _ in range(logic_output_dim):
        while True:
            select_idx = np.random.randint(len(train_X_true), size=samples_per_rule)
            X = []
            for idx in select_idx:
                X.append(train_X_true[idx])
            original_pred_res = model.predict(X)['cls']
            pred_res = mapping_res(original_pred_res, mapping)

            consistent_idx = []
            consistent_pred_res = []
            for idx in range(len(pred_res)):
                if abducer.kb.logic_forward([pred_res[idx]]):
                    consistent_idx.append(idx)
                    consistent_pred_res.append(pred_res[idx])

            if len(consistent_pred_res) != 0:
                rule = abducer.abduce_rules(consistent_pred_res)
                if rule != None:
                    break
        rules.append(rule)
    return rules


def get_mlp_vector(model, abducer, mapping, X, rules):
    original_pred_res = model.predict([X])['cls']
    pred_res = flatten(mapping_res(original_pred_res, mapping))
    vector = []
    for rule in rules:
        if abducer.kb.consist_rule(pred_res, rule):
            vector.append(1)
        else:
            vector.append(0)
    return vector


def get_mlp_data(model, abducer, mapping, X_true, X_false, rules):
    mlp_vectors = []
    mlp_labels = []
    for X in X_true:
        mlp_vectors.append(get_mlp_vector(model, abducer, mapping, X, rules))
        mlp_labels.append(1)
    for X in X_false:
        mlp_vectors.append(get_mlp_vector(model, abducer, mapping, X, rules))
        mlp_labels.append(0)
    return np.array(mlp_vectors, dtype=np.float32), np.array(mlp_labels, dtype=np.int64)

def get_all_mlp_data(model, abducer, mapping, X_true, X_false, rules, min_len, max_len):
    for equation_len in range(min_len, max_len + 1):
        mlp_vectors, mlp_labels = get_mlp_data(model, abducer, mapping, X_true[equation_len], X_false[equation_len], rules)
        if equation_len == min_len:
            all_mlp_vectors = mlp_vectors
            all_mlp_labels = mlp_labels
        else:
            all_mlp_vectors = np.concatenate((all_mlp_vectors, mlp_vectors))
            all_mlp_labels = np.concatenate((all_mlp_labels, mlp_labels))
    return all_mlp_vectors, all_mlp_labels


def validation(model, abducer, mapping, logic_output_dim, rules, train_X_true, train_X_false, val_X_true, val_X_false):
    mlp_train_vectors, mlp_train_labels = get_mlp_data(model, abducer, mapping, train_X_true, train_X_false, rules)
    mlp_train_data = BasicDataset(mlp_train_vectors, mlp_train_labels)
    
    mlp_val_vectors, mlp_val_labels = get_mlp_data(model, abducer, mapping, val_X_true, val_X_false, rules)
    mlp_val_data = BasicDataset(mlp_val_vectors, mlp_val_labels)
    
    best_accuracy = 0
    # Try three times to find the best mlp
    for _ in range(3):
        INFO("Training mlp...")
        mlp = MLP(input_dim=logic_output_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01, betas=(0.9, 0.999))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        mlp_model = BasicModel(mlp, criterion, optimizer, device, batch_size=128, num_epochs=100)
        mlp_train_data_loader = torch.utils.data.DataLoader(mlp_train_data, batch_size=128, shuffle=True)
        
        loss = mlp_model.fit(mlp_train_data_loader)
        INFO("mlp training final loss is %f" % loss)
        
        mlp_val_data_loader = torch.utils.data.DataLoader(mlp_val_data, batch_size=64, shuffle=True)
        accuracy = mlp_model.val(mlp_val_data_loader)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
    return best_accuracy






def train_with_rule(model, abducer, train_data, val_data, select_num=10, min_len=5, max_len=8):
    train_X = train_data
    val_X = val_data
    
    logic_output_dim = 50
    samples_per_rule = 3

    # Start training / for each length of equations
    for equation_len in range(min_len, max_len):
        INFO("============== equation_len: %d-%d ================" % (equation_len, equation_len + 1))
        train_X_true = train_X[1][equation_len]
        train_X_false = train_X[0][equation_len]
        val_X_true = val_X[1][equation_len]
        val_X_false = val_X[0][equation_len]
        
        train_X_true.extend(train_X[1][equation_len + 1])
        train_X_false.extend(train_X[0][equation_len + 1])
        val_X_true.extend(val_X[1][equation_len + 1])
        val_X_false.extend(val_X[0][equation_len + 1])

        condition_cnt = 0
        while True:
            if equation_len == min_len:
                mapping = None
            
            # Abduce and train NN
            consistent_acc, char_acc, mapping = abduce_and_train(model, abducer, mapping, train_X_true, select_num)
            if consistent_acc == 0:
                continue
            
            # Test if we can use mlp to evaluate
            if consistent_acc >= 0.9 and char_acc >= 0.9:
                condition_cnt += 1
            else:
                condition_cnt = 0

            # The condition has been satisfied continuously five times
            if condition_cnt >= 5:
                INFO("Now checking if we can go to next course")
                rules = get_rules_from_data(model, abducer, mapping, train_X_true, samples_per_rule, logic_output_dim)
                INFO('Learned rules from data:', output_rules(rules))
                best_accuracy = validation(model, abducer, mapping, logic_output_dim, rules, train_X_true, train_X_false, val_X_true, val_X_false)
                INFO('best_accuracy is %f\n' %(best_accuracy))
                # decide next course or restart
                if best_accuracy > 0.88:
                    torch.save(model.cls_list[0].model.state_dict(), "./weights/weights_%d.pth" % equation_len)
                    break
                else:
                    if equation_len == min_len:
                        model.cls_list[0].model.load_state_dict(torch.load("./weights/pretrain_weights.pth"))
                    else:
                        model.cls_list[0].model.load_state_dict(torch.load("./weights/weights_%d.pth" % (equation_len - 1)))
                    condition_cnt = 0
                    INFO('Reload Model and retrain')
                  
    return model, mapping

def hed_test(model, abducer, mapping, train_data, test_data, min_len=5, max_len=8):
    train_X = train_data
    test_X = test_data
    
    # Calcualte how many equations should be selected in each length
    # for each length, there are select_equation_cnt[equation_len] rules
    print("Now begin to train final mlp model")
    select_equation_cnt = []
    len_cnt = max_len - min_len + 1
    logic_output_dim = 50
    select_equation_cnt += [0] * min_len
    if logic_output_dim % len_cnt == 0:
        select_equation_cnt += [logic_output_dim // len_cnt] * len_cnt
    else:
        select_equation_cnt += [logic_output_dim // len_cnt] * len_cnt
        select_equation_cnt[-1] += logic_output_dim % len_cnt
    assert sum(select_equation_cnt) == logic_output_dim

    # Abduce rules
    rules = []
    samples_per_rule = 3
    for equation_len in range(min_len, max_len + 1):
        equation_rules = get_rules_from_data(model, abducer, mapping, train_X[1][equation_len], samples_per_rule, select_equation_cnt[equation_len])
        rules.extend(equation_rules)
    INFO('Learned rules from data:', output_rules(rules))
       
    mlp_train_vectors, mlp_train_labels = get_all_mlp_data(model, abducer, mapping, train_X[1], train_X[0], rules, min_len, max_len)
    mlp_train_data = BasicDataset(mlp_train_vectors, mlp_train_labels)
    
    # Try three times to find the best mlp
    for _ in range(3):
        mlp = MLP(input_dim=logic_output_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01, betas=(0.9, 0.999))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        mlp_model = BasicModel(mlp, criterion, optimizer, device, batch_size=128, num_epochs=100)
        mlp_train_data_loader = torch.utils.data.DataLoader(mlp_train_data, batch_size=128, shuffle=True)
        
        loss = mlp_model.fit(mlp_train_data_loader)
        INFO("mlp training final loss is %f" % loss)
        
        for equation_len in range(5, 27):
            mlp_test_vectors, mlp_test_labels = get_mlp_data(model, abducer, mapping, test_X[1][equation_len], test_X[0][equation_len], rules)
            mlp_test_data = BasicDataset(mlp_test_vectors, mlp_test_labels)
            mlp_test_data_loader = torch.utils.data.DataLoader(mlp_test_data, batch_size=64, shuffle=True)
            accuracy = mlp_model.val(mlp_test_data_loader)
            INFO("The accuracy of testing length %d equations is: %f" % (equation_len, accuracy))
        INFO("\n")

if __name__ == "__main__":
    pass
