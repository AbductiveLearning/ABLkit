# coding: utf-8
#================================================================#
#   Copyright (C) 2020 Freecss All rights reserved.
#   
#   File Name     ：plog.py
#   Author        ：freecss
#   Email         ：karlfreecss@gmail.com
#   Created Date  ：2020/10/23
#   Description   ：
#
#================================================================#

import time
import logging
import pickle as pk
import os
import functools

log_name = "default_log.txt"
logging.basicConfig(level=logging.INFO, 
    filename=log_name, 
    filemode='a', 
    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s') 

global recorder
recorder = None

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

class ResultRecorder:
    def __init__(self, pk_dir = None, pk_filepath = None):
        self.set_savefile(pk_dir, pk_filepath)

        self.result = {}
        logging.info("===========================================================")
        logging.info("============= Result Recorder Version: 0.02 ===============")
        logging.info("===========================================================\n")

        pass

    def set_savefile(self, pk_dir = None, pk_filepath = None):
        if pk_dir is None:
            pk_dir = "result"
        mkdir(pk_dir)

        if pk_filepath is None:
            local_time = time.strftime("%Y%m%d_%H_%M_%S", time.localtime()) 
            pk_filepath = os.path.join(pk_dir, local_time + ".pk")

        self.save_file = open(pk_filepath, "wb")
        
        logger = logging.getLogger()
        logger.handlers[0].stream.close()
        logger.removeHandler(logger.handlers[0])

        filename = os.path.join(pk_dir, local_time + ".txt")
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s') 
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def print(self, *argv, screen = False):
        info = ""
        for data in argv:
            info += str(data)
        if screen:
            print(info)
        logging.info(info)

    def print_result(self, *argv):
        for data in argv:
            info = "#Result# %s" % str(data)
            #print(info)
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
        self.print_result({label : data})
        #self.print_result(label + ":" + str(data))
        self.store_kv(label, data)

    def dump(self, save_file = None):
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
            context = f"{name}: ()=>(), cost {elapsed}s"
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

def INFO(*argv, screen = False):
    global recorder
    if recorder is None:
        recorder = ResultRecorder()
    return recorder.print(*argv, screen = screen)

def DEBUG(*argv, screen = False):
    global recorder
    if recorder is None:
        recorder = ResultRecorder()
    return recorder.print(*argv, screen = screen)

def logger():
    global recorder
    if recorder is None:
        recorder = ResultRecorder()
    return recorder
        
if __name__ == "__main__":
    recorder = ResultRecorder()
    recorder.write_kv("test", 1)
    recorder.set_savefile(pk_dir = "haha")
    recorder.write_kv("test", 1)

