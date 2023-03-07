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
from utils.utils import (
    flatten,
    reform_idx,
    block_sample,
    gen_mappings,
    mapping_res,
    remapping_res,
    extract_feature,
)

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


def hed_pretrain(cls, image_size=(28, 28, 1)):
    import cv2

    INFO("Pretrain Start")
    pretrain_data_X, pretrain_data_Y = [], []
    for i, label in enumerate(["0", "1", "10", "11"]):
        label_path = os.path.join("./datasets/hed/dataset/mnist_images", label)
        img_path_list = os.listdir(label_path)
        for j in range(10):
            img = cv2.imread(
                os.path.join(label_path, img_path_list[j]), cv2.IMREAD_GRAYSCALE
            )
            img = np.array(cv2.resize(img, (image_size[1], image_size[0])), np.float32)
            img = (img - 127) / 128.0
            pretrain_data_X.append(
                extract_feature(img.reshape((1, image_size[0], image_size[1])))
            )
            pretrain_data_Y.append(i)
    cls.fit(pretrain_data_X, pretrain_data_Y)
    import random

    for i, label in enumerate(["0", "1", "10", "11"]):
        label_path = os.path.join("./datasets/hed/dataset/mnist_images", label)
        img_path_list = os.listdir(label_path)
        cnt = 0
        for j in range(50):
            img = cv2.imread(
                os.path.join(label_path, random.choice(img_path_list)),
                cv2.IMREAD_GRAYSCALE,
            )
            img = np.array(cv2.resize(img, (image_size[1], image_size[0])), np.float32)
            img = (img - 127) / 128.0
            predict_label = cls.predict(
                [extract_feature(img.reshape((1, image_size[0], image_size[1])))]
            )
            # predict_label = cls.predict_proba(
            #     [
            #         extract_feature(
            #             np.array(img, dtype=np.float32).reshape(
            #                 (1, image_size[0], image_size[1])
            #             )
            #         )
            #     ]
            # ).argmax(axis=1)

            if predict_label == i:
                cnt += 1
        INFO(
            "%d predict accuracy is " % i,
            cnt / 50,
        )

    return pretrain_data_X, pretrain_data_Y


def _get_char_acc(model, X, consistent_pred_res, mapping):
    original_pred_res = model.predict(X)["cls"]
    pred_res = flatten(mapping_res(original_pred_res, mapping))
    INFO("Current model's output: ", pred_res)
    INFO("Abduced labels:         ", flatten(consistent_pred_res))
    assert len(pred_res) == len(flatten(consistent_pred_res))
    return sum(
        [
            pred_res[idx] == flatten(consistent_pred_res)[idx]
            for idx in range(len(pred_res))
        ]
    ) / len(pred_res)


def abduce_and_train(model, abducer, mapping, train_X_true, pretrain_data, select_num):
    select_idx = np.random.randint(len(train_X_true), size=select_num)
    X = []
    for idx in select_idx:
        X.append(train_X_true[idx])

    original_pred_res = model.predict(X)["cls"]

    if mapping == None:
        mappings = gen_mappings(["+", "=", 0, 1], ["+", "=", 0, 1])
    else:
        mappings = [mapping]

    consistent_idx = []
    consistent_pred_res = []

    for m in mappings:
        pred_res = mapping_res(original_pred_res, m)
        max_abduce_num = 20
        solution = abducer.zoopt_get_solution(
            pred_res, [1] * len(pred_res), max_abduce_num
        )
        all_address_flag = reform_idx(solution, pred_res)

        consistent_idx_tmp = []
        consistent_pred_res_tmp = []

        for idx in range(len(pred_res)):
            address_idx = [
                i for i, flag in enumerate(all_address_flag[idx]) if flag != 0
            ]
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
        INFO("Final mapping is: ", mapping)

    INFO("Train pool size is:", len(flatten(consistent_pred_res)))
    INFO("Start to use abduced pseudo label to train model...")
    pretrain_data_X, pretrain_data_Y = pretrain_data
    pretrain_mappping = {0: 0, 1: 1, 2: "+", 3: "="}
    pretrain_data_X = [[X] for X in pretrain_data_X]
    pretrain_data_Y = [[pretrain_mappping[Y]] for Y in pretrain_data_Y]
    model.train(
        [X[idx] for idx in consistent_idx] + pretrain_data_X,
        remapping_res(consistent_pred_res + pretrain_data_Y, mapping),
    )

    consistent_acc = len(consistent_idx) / select_num
    char_acc = _get_char_acc(
        model, [X[idx] for idx in consistent_idx], consistent_pred_res, mapping
    )
    INFO("consistent_acc is %s, char_acc is %s" % (consistent_acc, char_acc))
    return consistent_acc, char_acc, mapping


def _remove_duplicate_rule(rule_dict):
    add_nums_dict = {}
    for r in list(rule_dict):
        add_nums = str(r.split("]")[0].split("[")[1]) + str(
            r.split("]")[1].split("[")[1]
        )  # r = 'my_op([1], [0], [1, 0])' then add_nums = '10'
        if add_nums in add_nums_dict:
            old_r = add_nums_dict[add_nums]
            if rule_dict[r] >= rule_dict[old_r]:
                rule_dict.pop(old_r)
                add_nums_dict[add_nums] = r
            else:
                rule_dict.pop(r)
        else:
            add_nums_dict[add_nums] = r
    return list(rule_dict)


def get_rules_from_data(
    model, abducer, mapping, train_X_true, samples_per_rule, samples_num
):
    rules = []
    for _ in range(samples_num):
        while True:
            select_idx = np.random.randint(len(train_X_true), size=samples_per_rule)
            X = []
            for idx in select_idx:
                X.append(train_X_true[idx])
            original_pred_res = model.predict(X)["cls"]
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

    all_rule_dict = {}
    for rule in rules:
        for r in rule:
            all_rule_dict[r] = 1 if r not in all_rule_dict else all_rule_dict[r] + 1
    rule_dict = {rule: cnt for rule, cnt in all_rule_dict.items() if cnt >= 5}
    rules = _remove_duplicate_rule(rule_dict)

    return rules


def _get_consist_rule_acc(model, abducer, mapping, rules, X):
    cnt = 0
    for x in X:
        original_pred_res = model.predict([x])["cls"]
        pred_res = flatten(mapping_res(original_pred_res, mapping))
        if abducer.kb.consist_rule(pred_res, rules):
            cnt += 1
    return cnt / len(X)


def train_with_rule(
    model,
    abducer,
    train_data,
    val_data,
    pretrain_data,
    select_num=10,
    min_len=5,
    max_len=8,
):
    train_X = train_data
    val_X = val_data

    samples_num = 50
    samples_per_rule = 3

    # Start training / for each length of equations
    for equation_len in range(min_len, max_len):
        INFO(
            "============== equation_len: %d-%d ================"
            % (equation_len, equation_len + 1)
        )
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
            consistent_acc, char_acc, mapping = abduce_and_train(
                model, abducer, mapping, train_X_true, pretrain_data, select_num
            )
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
                rules = get_rules_from_data(
                    model, abducer, mapping, train_X_true, samples_per_rule, samples_num
                )
                INFO("Learned rules from data:", rules)

                true_consist_rule_acc = _get_consist_rule_acc(
                    model, abducer, mapping, rules, val_X_true
                )
                false_consist_rule_acc = _get_consist_rule_acc(
                    model, abducer, mapping, rules, val_X_false
                )

                INFO(
                    "consist_rule_acc is %f, %f\n"
                    % (true_consist_rule_acc, false_consist_rule_acc)
                )
                # decide next course or restart
                if true_consist_rule_acc > 0.9 and false_consist_rule_acc < 0.1:
                    break
                else:
                    if equation_len == min_len:
                        # model.cls_list[0].model.load_state_dict(
                        #     torch.load("./weights/pretrain_weights.pth")
                        # )
                        pretrain_data_X, pretrain_data_Y = pretrain_data
                        model.cls_list[0].fit(pretrain_data_X, pretrain_data_Y)
                    else:
                        pretrain_data_X, pretrain_data_Y = pretrain_data
                        model.cls_list[0].fit(pretrain_data_X, pretrain_data_Y)
                        # model.cls_list[0].model.load_state_dict(
                        #     torch.load("./weights/weights_%d.pth" % (equation_len - 1))
                        # )
                    condition_cnt = 0
                    INFO("Reload Model and retrain")

    return model, mapping


def hed_test(model, abducer, mapping, train_data, test_data, min_len=5, max_len=8):
    train_X = train_data
    test_X = test_data

    # Calcualte how many equations should be selected in each length
    # for each length, there are equation_samples_num[equation_len] rules
    print("Now begin to train final mlp model")
    equation_samples_num = []
    len_cnt = max_len - min_len + 1
    samples_num = 50
    equation_samples_num += [0] * min_len
    if samples_num % len_cnt == 0:
        equation_samples_num += [samples_num // len_cnt] * len_cnt
    else:
        equation_samples_num += [samples_num // len_cnt] * len_cnt
        equation_samples_num[-1] += samples_num % len_cnt
    assert sum(equation_samples_num) == samples_num

    # Abduce rules
    rules = []
    samples_per_rule = 3
    for equation_len in range(min_len, max_len + 1):
        equation_rules = get_rules_from_data(
            model,
            abducer,
            mapping,
            train_X[1][equation_len],
            samples_per_rule,
            equation_samples_num[equation_len],
        )
        rules.extend(equation_rules)
    rules = list(set(rules))
    INFO("Learned rules from data:", rules)

    for equation_len in range(5, 27):
        true_consist_rule_acc = _get_consist_rule_acc(
            model, abducer, mapping, rules, test_X[1][equation_len]
        )
        false_consist_rule_acc = _get_consist_rule_acc(
            model, abducer, mapping, rules, test_X[0][equation_len]
        )
        INFO(
            "consist_rule_acc of testing length %d equations are %f, %f"
            % (equation_len, true_consist_rule_acc, false_consist_rule_acc)
        )


if __name__ == "__main__":
    pass
