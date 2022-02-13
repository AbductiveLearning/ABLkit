# coding: utf-8
#================================================================#
#   Copyright (C) 2020 Freecss All rights reserved.
#   
#   File Name     ：basic_model.py
#   Author        ：freecss
#   Email         ：karlfreecss@gmail.com
#   Created Date  ：2020/11/21
#   Description   ：
#
#================================================================#

import sys
sys.path.append("..")

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import torchvision

import utils.utils as mutils

import os
from multiprocessing import Pool

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import six
import sys
from PIL import Image
import numpy as np
import collections

class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.transform = transforms.Compose([
            #transforms.ToPILImage(),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            #transforms.RandomRotation(30),
            #transforms.RandomAffine(30),
            transforms.ToTensor(),

        ])

    def __call__(self, img):
        #img = img.resize(self.size, self.interpolation)
        #img = self.toTensor(img)
        img = self.transform(img)
        img.sub_(0.5).div_(0.5)
        return img

class XYDataset(Dataset):
    def __init__(self, X, Y, transform=None, target_transform=None):
        self.X = X
        self.Y = Y

        self.n_sample = len(X)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        assert index < len(self), 'index range error'
        
        img = self.X[index]
        if self.transform is not None:
            img = self.transform(img)

        label = self.Y[index]
        if self.target_transform is not None:
            label = self.target_transform(label)

        return (img, label, index)

class alignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels, img_keys = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.shape[:2]
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)
        labels = torch.LongTensor(labels)

        return images, labels, img_keys

class FakeRecorder():
    def __init__(self):
        pass

    def print(self, *x):
        pass

from torch.nn import init
from torch import nn
def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data,0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0,0.01)
        m.bias.data.zero_()

class BasicModel():
    def __init__(self,
            model,
            criterion,
            optimizer,
            converter,
            device,
            params,
            sign_list,
            recorder = None):

        self.model = model.to(device)
        self.model.apply(weigth_init)
        self.criterion = criterion
        self.optimizer = optimizer
        self.converter = converter
        self.device = device
        sign_list = sorted(list(set(sign_list)))
        self.mapping = dict(zip(sign_list, list(range(len(sign_list)))))
        self.remapping = dict(zip(list(range(len(sign_list))), sign_list))

        if recorder is None:
            recorder = FakeRecorder()
        self.recorder = recorder

        self.save_interval = params.saveInterval
        self.params = params
        pass

    def _fit(self, data_loader, n_epoch, stop_loss):
        recorder = self.recorder
        recorder.print("model fitting")

        min_loss = 999999999
        for epoch in range(n_epoch):
            loss_value = self.train_epoch(data_loader)
            recorder.print(f"{epoch}/{n_epoch} model training loss is {loss_value}")
            if loss_value < min_loss:
                min_loss = loss_value
            if loss_value < stop_loss:
                break
        recorder.print("Model fitted, minimal loss is ", min_loss)
        return loss_value

    def str2ints(self, Y):
        return [self.mapping[y] for y in Y]

    def fit(self, data_loader = None,
                  X = None,
                  y = None,
                  n_epoch = 100,
                  stop_loss = 0.001):
        if data_loader is None:
            params = self.params
            Y = self.str2ints(y)
            train_dataset = XYDataset(X, Y)
            sampler = None
            data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batchSize, \
                    shuffle=True, sampler=sampler, num_workers=int(params.workers), \
                    collate_fn=alignCollate(imgH=params.imgH, imgW=params.imgW, keep_ratio=params.keep_ratio))
        return self._fit(data_loader, n_epoch, stop_loss)

    def train_epoch(self, data_loader):
        loss_avg = mutils.averager()

        for i, data in enumerate(data_loader):
            X = data[0]
            Y = data[1]
            cost = self.train_batch(X, Y)
            loss_avg.add(cost)
    
        loss_value = float(loss_avg.val())
        loss_avg.reset()
        return loss_value

    def train_batch(self, X, Y):
        #cpu_images, cpu_texts, _ = data
        model = self.model
        criterion = self.criterion
        optimizer = self.optimizer
        converter = self.converter
        device = self.device

        # set training mode
        for p in model.parameters():
            p.requires_grad = True
        model.train()
    
        # init training status
        torch.autograd.set_detect_anomaly(True)
        optimizer.zero_grad()

        # model predict
        X = X.to(device)
        Y = Y.to(device)
        pred_Y = model(X)

        # calculate loss
        loss = criterion(pred_Y, Y)

        # back propagation and optimize
        loss.backward()
        optimizer.step()
        return loss

    def _predict(self, data_loader):
        model = self.model
        criterion = self.criterion
        converter = self.converter
        params = self.params
        device = self.device
        
        for p in model.parameters():
            p.requires_grad = False
    
        model.eval()
    
        n_correct = 0
    
        results = []
        for i, data in enumerate(data_loader):
            X = data[0].to(device)
            pred_Y = model(X)
            results.append(pred_Y)
    
        return torch.cat(results, axis=0)

    def predict(self, data_loader = None, X = None, print_prefix = ""):
        params = self.params
        if data_loader is None:
            Y = [0] * len(X)
            val_dataset = XYDataset(X, Y)
            sampler = None
            data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=params.batchSize, \
                shuffle=False, sampler=sampler, num_workers=int(params.workers), \
                collate_fn=alignCollate(imgH=params.imgH, imgW=params.imgW, keep_ratio=params.keep_ratio))

        recorder = self.recorder
        recorder.print('Start Predict ', print_prefix)
        Y = self._predict(data_loader).argmax(axis=1)
        return [self.remapping[int(y)] for y in Y]

    def predict_proba(self, data_loader = None, X = None, print_prefix = ""):
        params = self.params
        if data_loader is None:
            Y = [0] * len(X)
            val_dataset = XYDataset(X, Y)
            sampler = None
            data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=params.batchSize, \
                shuffle=False, sampler=sampler, num_workers=int(params.workers), \
                collate_fn=alignCollate(imgH=params.imgH, imgW=params.imgW, keep_ratio=params.keep_ratio))

        recorder = self.recorder
        recorder.print('Start Predict ', print_prefix)
        return torch.softmax(self._predict(data_loader), axis=1)

    def _val(self, data_loader, print_prefix):
        model = self.model
        criterion = self.criterion
        recorder = self.recorder
        converter = self.converter
        params = self.params
        device = self.device
        recorder.print('Start val ', print_prefix)
        
        for p in model.parameters():
            p.requires_grad = False
    
        model.eval()
    
        n_correct = 0
        pred_num = 0
        loss_avg = mutils.averager()
        for i, data in enumerate(data_loader):
            X = data[0].to(device)
            Y = data[1].to(device)

            pred_Y = model(X)

            correct_num = sum(Y == pred_Y.argmax(axis=1))
            loss = criterion(pred_Y, Y)
            loss_avg.add(loss)

            n_correct += correct_num
            pred_num += len(X)

        accuracy = float(n_correct) / float(pred_num)
        recorder.print('[%s] Val loss: %f, accuray: %f' % (print_prefix, loss_avg.val(), accuracy))
        return accuracy

    def val(self, data_loader = None, X = None, y = None, print_prefix = ""):
        params = self.params
        if data_loader is None:
            y = self.str2ints(y)
            val_dataset = XYDataset(X, y)
            sampler = None
            data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=params.batchSize, \
                shuffle=True, sampler=sampler, num_workers=int(params.workers), \
                collate_fn=alignCollate(imgH=params.imgH, imgW=params.imgW, keep_ratio=params.keep_ratio))
        return self._val(data_loader, print_prefix)

    def score(self, data_loader = None, X = None, y = None, print_prefix = ""):
        return self.val(data_loader, X, y, print_prefix)

    def save(self, save_dir):
        recorder = self.recorder
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        recorder.print("Saving model and opter")
        save_path = os.path.join(save_dir, "net.pth")
        torch.save(self.model.state_dict(), save_path)

        save_path = os.path.join(save_dir, "opt.pth")
        torch.save(self.optimizer.state_dict(), save_path)

    def load(self, load_dir):
        recorder = self.recorder
        recorder.print("Loading model and opter")
        load_path = os.path.join(load_dir, "net.pth")
        self.model.load_state_dict(torch.load(load_path))

        load_path = os.path.join(load_dir, "opt.pth")
        self.optimizer.load_state_dict(torch.load(load_path))

if __name__ == "__main__":
    pass


