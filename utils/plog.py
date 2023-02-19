# coding: utf-8
# ================================================================#
#   Copyright (C) 2020 Freecss All rights reserved.
#
#   File Name     ：plog.py
#   Author        ：freecss
#   Email         ：karlfreecss@gmail.com
#   Created Date  ：2020/10/23
#   Description   ：
#
# ================================================================#

import time
import logging
import pickle as pk
import os
import functools

global recorder
recorder = None


class ResultRecorder:
    def __init__(self):
        logging.basicConfig(level=logging.DEBUG, filemode="a")

        self.result = {}
        self.set_savefile()

        logging.info("===========================================================")
        logging.info("============= Result Recorder Version: 0.03 ===============")
        logging.info("===========================================================\n")

        pass

    def set_savefile(self):
        local_time = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())

        save_dir = os.path.join("results", local_time)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_file_path = os.path.join(save_dir, "result.pk")
        save_file = open(save_file_path, "wb")

        self.save_dir = save_dir
        self.save_file = save_file

        filename = os.path.join(save_dir, "log.txt")
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)

    def print(self, *argv, screen=False):
        info = ""
        for data in argv:
            info += str(data)
        if screen:
            print(info)
        logging.info(info)

    def print_result(self, *argv):
        for data in argv:
            info = "#Result# %s" % str(data)
            logging.info(info)

    def store(self, *argv):
        for data in argv:
            if data.find(":") < 0:
                continue
            label, data = data.split(":")
            self.store_kv(label, data)

    def write_result(self, *argv):
        self.print_result(*argv)
        self.store(*argv)

    def store_kv(self, label, data):
        self.result.setdefault(label, [])
        self.result[label].append(data)

    def write_kv(self, label, data):
        self.print_result({label: data})
        # self.print_result(label + ":" + str(data))
        self.store_kv(label, data)

    def dump(self, save_file=None):
        if save_file is None:
            save_file = self.save_file
        pk.dump(self.result, save_file)

    def clock(self, func):
        @functools.wraps(func)
        def clocked(*args, **kwargs):
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - t0

            name = func.__name__
            # arg_str = ','.join(repr(arg) for arg in args)
            # context = f"{name}: ({arg_str})=>({result}), cost {elapsed}s"
            context = f"{name}: cost {elapsed}s"
            self.write_kv("func:", context)

            return result

        return clocked

    def __del__(self):
        self.dump()


def clocker(*argv):
    global recorder
    if recorder is None:
        recorder = ResultRecorder()
    return recorder.clock(*argv)


def INFO(*argv, screen=False):
    global recorder
    if recorder is None:
        recorder = ResultRecorder()
    return recorder.print(*argv, screen=screen)


def DEBUG(*argv, screen=False):
    global recorder
    if recorder is None:
        recorder = ResultRecorder()
    return recorder.print(*argv, screen=screen)


def logger():
    global recorder
    if recorder is None:
        recorder = ResultRecorder()
    return recorder


if __name__ == "__main__":
    recorder = ResultRecorder()
    recorder.write_kv("test", 1)
    recorder.set_savefile(pk_dir="haha")
    recorder.write_kv("test", 1)
