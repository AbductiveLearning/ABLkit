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
from torch.utils.data import Dataset

import os
from multiprocessing import Pool

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

class FakeRecorder():
    def __init__(self):
        pass

    def print(self, *x):
        pass

class BasicModel():
    def __init__(self,
            model,
            criterion,
            optimizer,
            device,
            params,
            transform = None,
            target_transform=None,
            collate_fn = None,
            recorder = None):

        self.model = model.to(device)
        
        self.criterion = criterion
        self.optimizer = optimizer
        self.transform = transform
        self.target_transform = target_transform
        self.device = device

        if recorder is None:
            recorder = FakeRecorder()
        self.recorder = recorder

        self.save_interval = params.saveInterval
        self.params = params
        self.collate_fn = collate_fn
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
            if epoch > 0 and self.save_interval is not None and epoch % self.save_interval == 0:
                assert hasattr(self.params, 'save_dir')
                self.save(self.params.save_dir)
            if stop_loss is not None and  loss_value < stop_loss:
                break
        recorder.print("Model fitted, minimal loss is ", min_loss)
        return loss_value

    def fit(self, data_loader = None,
                  X = None,
                  y = None):
        if data_loader is None:
            params = self.params
            collate_fn = self.collate_fn
            transform = self.transform
            target_transform = self.target_transform

            train_dataset = XYDataset(X, y, transform=transform, target_transform=target_transform)
            sampler = None
            data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batchSize, \
                    shuffle=True, sampler=sampler, num_workers=int(params.workers), \
                    collate_fn=collate_fn)
        return self._fit(data_loader, params.n_epoch, params.stop_loss)

    def train_epoch(self, data_loader):
        model = self.model
        criterion = self.criterion
        optimizer = self.optimizer
        device = self.device
        
        model.train()

        loss_value = 0
        for _, data in enumerate(data_loader):
            X = data[0].to(device)
            Y = data[1].to(device)
            pred_Y = model(X)

            loss = criterion(pred_Y, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value += loss.item()

        return loss_value

    def _predict(self, data_loader):
        model = self.model
        device = self.device
    
        model.eval()

        with torch.no_grad():
            results = []
            for _, data in enumerate(data_loader):
                X = data[0].to(device)
                pred_Y = model(X)
                results.append(pred_Y)
    
        return torch.cat(results, axis=0)

    def predict(self, data_loader = None, X = None, print_prefix = ""):
        if data_loader is None:
            params = self.params
            collate_fn = self.collate_fn
            transform = self.transform
            target_transform = self.target_transform

            Y = [0] * len(X)
            val_dataset = XYDataset(X, Y, transform=transform, target_transform=target_transform)
            sampler = None
            data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=params.batchSize, \
                shuffle=False, sampler=sampler, num_workers=int(params.workers), \
                collate_fn=collate_fn)

        recorder = self.recorder
        recorder.print('Start Predict ', print_prefix)
        Y = self._predict(data_loader).argmax(axis=1)
        return [int(y) for y in Y]

    def predict_proba(self, data_loader = None, X = None, print_prefix = ""):
        if data_loader is None:
            params = self.params
            collate_fn = self.collate_fn
            transform = self.transform
            target_transform = self.target_transform

            Y = [0] * len(X)
            val_dataset = XYDataset(X, Y, transform=transform, target_transform=target_transform)
            sampler = None
            data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=params.batchSize, \
                shuffle=False, sampler=sampler, num_workers=int(params.workers), \
                collate_fn=collate_fn)

        recorder = self.recorder
        recorder.print('Start Predict ', print_prefix)
        return torch.softmax(self._predict(data_loader), axis=1).cpu().numpy()

    def _val(self, data_loader, print_prefix):
        model = self.model
        criterion = self.criterion
        recorder = self.recorder
        device = self.device
        recorder.print('Start val ', print_prefix)
    
        model.eval()
    
        n_correct = 0
        pred_num = 0
        loss_value = 0
        with torch.no_grad():
            for _, data in enumerate(data_loader):
                X = data[0].to(device)
                Y = data[1].to(device)

                pred_Y = model(X)

                correct_num = sum(Y == pred_Y.argmax(axis=1))
                loss = criterion(pred_Y, Y)
                loss_value += loss.item()

            n_correct += correct_num
            pred_num += len(X)

        accuracy = float(n_correct) / float(pred_num)
        recorder.print('[%s] Val loss: %f, accuray: %f' % (print_prefix, loss_value, accuracy))
        return accuracy

    def val(self, data_loader = None, X = None, y = None, print_prefix = ""):
        if data_loader is None:
            params = self.params
            collate_fn = self.collate_fn
            transform = self.transform
            target_transform = self.target_transform

            val_dataset = XYDataset(X, y, transform=transform, target_transform=target_transform)
            sampler = None
            data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=params.batchSize, \
                shuffle=True, sampler=sampler, num_workers=int(params.workers), \
                collate_fn=collate_fn)
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


