# coding: utf-8
#================================================================#
#   Copyright (C) 2020 Freecss All rights reserved.
#   
#   File Name     ：data_generator.py
#   Author        ：freecss
#   Email         ：karlfreecss@gmail.com
#   Created Date  ：2020/04/02
#   Description   ：
#
#================================================================#

from itertools import product
import math
import numpy as np
import random
import pickle as pk
import random
from multiprocessing import Pool
import copy

#def hamming_code_generator(data_len, p_len):
#    ret = []
#    for data in product((0, 1), repeat=data_len):
#        p_idxs = [2 ** i for i in range(p_len)]
#        total_len = data_len + p_len
#        data_idx = 0
#        hamming_code = []
#        for idx in range(total_len):
#            if idx + 1 in p_idxs:
#                hamming_code.append(0)
#            else:
#                hamming_code.append(data[data_idx])
#                data_idx += 1
#
#        for idx in range(total_len):
#            if idx + 1 in p_idxs:
#                for i in range(total_len):
#                    if (i + 1) & (idx + 1) != 0:
#                        hamming_code[idx] ^= hamming_code[i]
#        #hamming_code = "".join([str(x) for x in hamming_code])
#        ret.append(hamming_code)
#    return ret

def code_generator(code_len, code_num, letter_num = 2):
    codes = list(product(list(range(letter_num)), repeat = code_len))
    random.shuffle(codes)
    return codes[:code_num]

def hamming_distance_static(codes):
    min_dist = len(codes)
    avg_dist = 0.
    avg_min_dist = 0.
    relation_num = 0.
    for code1 in codes:
        tmp_min_dist = len(codes)
        for code2 in codes:
            if code1 == code2:
                continue
            dist = 0
            relation_num += 1
            for c1, c2 in zip(code1, code2):
                if c1 != c2:
                    dist += 1
            avg_dist += dist
            if tmp_min_dist > dist:
                tmp_min_dist = dist
        avg_min_dist += tmp_min_dist
        if min_dist > tmp_min_dist:
            min_dist = tmp_min_dist
    return avg_dist / relation_num, avg_min_dist / len(codes)

def generate_cosin_data(codes, err, repeat, letter_num):
    Y = np.random.random(100000) * letter_num * 3 - 3
    X = np.random.random(100000) * 20 - 10
    data_X = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis = 1)

    samples = {}
    all_sign = list(set(sum([[c for c in code] for code in codes], [])))

    for d, sign in enumerate(all_sign):
        labels = np.logical_and(Y < np.cos(X) + 2 * d, Y > np.cos(X) + 2 * d - 2)
        samples[sign] = data_X[labels]

    data = []
    labels = []
    count = 0
    for _ in range(repeat):
        if (count > 100000):
            break
        for code in codes:
            tmp = []
            count += 1
            for d in code:
                if random.random() < err:
                    candidates = copy.deepcopy(all_sign)
                    candidates.remove(d)
                    d = candidates[random.randint(0, letter_num - 2)]
                idx = random.randint(0, len(samples[d]) - 1)
                tmp.append(samples[d][idx])
            data.append(tmp)
            labels.append(code)
    data = np.array(data)
    labels = np.array(labels)
    return data, labels


#codes = """110011001
#100011001
#101101101
#011111001
#100100001
#111111101
#101110001
#111100101
#101000101
#001001101
#111110101
#100101001
#010010101
#110100101
#001111101
#111111001"""
#codes = codes.split()

def generate_data_via_codes(codes, err, letter_num):
    #codes = code_generator(code_len, code_num)
    data, labels = generate_cosin_data(codes, err, 100000, letter_num)
    return data, labels

def generate_data(params):
    code_len = params["code_len"]
    times = params["times"]
    p = params["p"]
    code_num = params["code_num"]

    err = p / 20.
    codes = code_generator(code_len, code_num)
    data, labels = generate_cosin_data(codes, err)
    data_name = "code_%d_%d" % (code_len, code_num)
    pk.dump((codes, data, labels), open("generated_data/%d_%s_%.2f.pk" % (times, data_name, err), "wb"))
    return True

def generate_multi_data():
    pool = Pool(64)
    params_list = []
    #for code_len in [7, 9, 11, 13, 15]:
    for code_len in [7, 11, 15]:
        for times in range(20):
            for p in range(0, 11):
                for code_num_power in range(1, code_len):
                    code_num = 2 ** code_num_power
                    params_list.append({"code_len" : code_len, "times" : times, "p" : p, "code_num" : code_num})
    return list(pool.map(generate_data, params_list))

def read_lexicon(file_path):
    ret = []
    with open(file_path) as fin:
        ret = [s.strip() for s in fin]

    all_sign = list(set(sum([[c for c in s] for s in ret], [])))
    #ret = ["".join(str(all_sign.index(t)) for t in tmp) for tmp in ret]

    return ret, len(all_sign)

import os

if __name__ == "__main__":
    for root, dirs, files in os.walk("lexicons"):
        if root != "lexicons":
            continue
        for file_name in files:
            file_path = os.path.join(root, file_name)
            codes, letter_num = read_lexicon(file_path)
            data, labels = generate_data_via_codes(codes, 0, letter_num)

            save_path = os.path.join("dataset", file_name.split(".")[0] + ".pk")
            pk.dump((data, labels, codes), open(save_path, "wb"))
            

    #res = read_lexicon("add2.txt")
    #print(res)
    exit(0)

    generate_multi_data()
    exit()
