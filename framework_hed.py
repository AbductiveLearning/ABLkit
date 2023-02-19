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

import numpy as np

import random

random_seed = random.randint(0, 10000)
print("Selected random seed is : ", random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

from models.nn import MLP
from models.basic_model import BasicModel, BasicDataset
import torch.nn as nn
import torch

from utils.plog import INFO, DEBUG, clocker
from utils.utils import flatten, reform_idx, block_sample

from sklearn.tree import DecisionTreeClassifier


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


def pretrain(net, pretrain_data_loader, recorder):
    INFO("Pretrain Start")

    criterion = nn.MSELoss()
    optimizer = torch.optim.RMSprop(
        net.parameters(), lr=0.001, alpha=0.9, weight_decay=1e-6
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pretrain_model = BasicModel(
        net,
        criterion,
        optimizer,
        device,
        save_interval=1,
        save_dir=recorder.save_dir,
        num_epochs=10,
        recorder=recorder,
    )

    pretrain_model.fit(pretrain_data_loader)


def get_char_acc(model, X, consistent_pred_res):
    pred_res = flatten(model.predict(X)["cls"])
    assert len(pred_res) == len(flatten(consistent_pred_res))
    return sum(
        [
            pred_res[idx] == flatten(consistent_pred_res)[idx]
            for idx in range(len(pred_res))
        ]
    ) / len(pred_res)


def gen_mappings(chars, symbs):
    n_char = len(chars)
    n_symbs = len(symbs)
    if n_char != n_symbs:
        INFO("Characters and symbols size dosen't match.")
        return
    from itertools import permutations

    mappings = []
    perms = permutations(symbs)
    for p in perms:
        mappings.append(dict(zip(chars, list(p))))
    return mappings


def map_res(original_pred_res, m):
    pred_res = [[m[symbol] for symbol in formula] for formula in original_pred_res]
    return pred_res


def abduce_and_train(model, abducer, train_X_true, select_num):
    select_idx = np.random.randint(len(train_X_true), size=select_num)
    X = []
    for idx in select_idx:
        X.append(train_X_true[idx])

    pred_res = model.predict(X)["cls"]

    maps = gen_mappings(["+", "=", 0, 1], ["+", "=", 0, 1])

    consistent_idx = []
    consistent_pred_res = []

    import copy

    original_pred_res = copy.deepcopy(pred_res)
    mapping = None

    for m in maps:
        pred_res = map_res(original_pred_res, m)
        remapping = {}
        for key, value in m.items():
            remapping[value] = key

        max_abduce_num = 10
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
                consistent_pred_res_tmp.append(
                    [remapping[symbol] for symbol in candidate[0][0]]
                )

        if len(consistent_idx_tmp) > len(consistent_idx):
            consistent_idx = consistent_idx_tmp
            consistent_pred_res = consistent_pred_res_tmp
            mapping = m

    if len(consistent_idx) == 0:
        return 0, 0, None

    INFO("Consistent predict results are:", map_res(consistent_pred_res, mapping))
    INFO("Train pool size is:", len(flatten(consistent_pred_res)))
    INFO("Start to use abduced pseudo label to train model...")
    model.train([X[idx] for idx in consistent_idx], consistent_pred_res)

    consistent_acc = len(consistent_idx) / select_num
    char_acc = get_char_acc(
        model, [X[idx] for idx in consistent_idx], consistent_pred_res
    )
    INFO("consistent_acc is %s, char_acc is %s" % (consistent_acc, char_acc))
    return consistent_acc, char_acc, mapping


def get_rules_from_data(
    model, abducer, mapping, train_X_true, samples_per_rule, logic_output_dim
):
    rules = []
    for _ in range(logic_output_dim):
        while True:
            select_idx = np.random.randint(len(train_X_true), size=samples_per_rule)
            X = []
            for idx in select_idx:
                X.append(train_X_true[idx])
            pred_res = model.predict(X)["cls"]
            pred_res = [[mapping[symbol] for symbol in formula] for formula in pred_res]

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
    INFO('Learned rules from data:')
    for rule in rules:
        INFO(rule)
    return rules


def get_mlp_vector(model, abducer, mapping, X, rules):
    pred_res = model.predict([X])["cls"]
    pred_res = [[mapping[symbol] for symbol in formula] for formula in pred_res]
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


def validation(
    model,
    abducer,
    mapping,
    train_X_true,
    train_X_false,
    val_X_true,
    val_X_false,
    recorder,
):
    INFO("Now checking if we can go to next course")
    samples_per_rule = 3
    logic_output_dim = 50
    rules = get_rules_from_data(
        model, abducer, mapping, train_X_true, samples_per_rule, logic_output_dim
    )

    mlp_train_vectors, mlp_train_labels = get_mlp_data(
        model, abducer, mapping, train_X_true, train_X_false, rules
    )

    idx = np.array(list(range(len(mlp_train_labels))))
    np.random.shuffle(idx)
    mlp_train_vectors = mlp_train_vectors[idx]
    mlp_train_labels = mlp_train_labels[idx]

    best_accuracy = 0

    # Try three times to find the best mlp
    for _ in range(3):
        INFO("Training mlp...")
        mlp = MLP(input_dim=logic_output_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01, betas=(0.9, 0.999))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        mlp_model = BasicModel(
            mlp,
            criterion,
            optimizer,
            device,
            batch_size=128,
            num_epochs=60,
            recorder=recorder,
        )
        mlp_train_data = BasicDataset(mlp_train_vectors, mlp_train_labels)
        mlp_train_data_loader = torch.utils.data.DataLoader(
            mlp_train_data,
            batch_size=128,
            shuffle=True,
        )
        loss = mlp_model.fit(mlp_train_data_loader)
        INFO("mlp training loss is %f" % loss)

        mlp_val_vectors, mlp_val_labels = get_mlp_data(
            model, abducer, mapping, val_X_true, val_X_false, rules
        )

        # Get MLP validation result
        mlp_val_data = BasicDataset(mlp_val_vectors, mlp_val_labels)
        mlp_val_data_loader = torch.utils.data.DataLoader(
            mlp_val_data,
            batch_size=64,
            shuffle=True,
        )

        accuracy = mlp_model.val(mlp_val_data_loader)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
    return best_accuracy


def train_with_rule(
    model,
    abducer,
    train_data,
    val_data,
    select_num=10,
    recorder=None
):
    train_X = train_data
    val_X = val_data

    min_len = 5
    max_len = 8

    # Start training / for each length of equations
    for equation_len in range(min_len, max_len):
        INFO("============== equation_len:%d ================" % (equation_len))
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
            # Abduce and train NN
            consistent_acc, char_acc, mapping = abduce_and_train(
                model, abducer, train_X_true, select_num
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
                best_accuracy = validation(
                    model,
                    abducer,
                    mapping,
                    train_X_true,
                    train_X_false,
                    val_X_true,
                    val_X_false,
                    recorder,
                )

                INFO("best_accuracy is %f" % (best_accuracy))

                # decide next course or restart
                if best_accuracy > 0.86:
                    torch.save(
                        model.cls_list[0].model.state_dict(),
                        "./weights/train_weights_%d.pth" % equation_len,
                    )
                    break
                else:
                    if equation_len == min_len:
                        model.cls_list[0].model.load_state_dict(
                            torch.load("./weights/pretrain_weights.pth")
                        )
                    else:
                        model.cls_list[0].model.load_state_dict(
                            torch.load(
                                "./weights/train_weights_%d.pth" % (equation_len - 1)
                            )
                        )
                    condition_cnt = 0

    return model


if __name__ == "__main__":
    pass
