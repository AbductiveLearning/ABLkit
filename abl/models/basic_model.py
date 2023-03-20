# coding: utf-8
# ================================================================#
#   Copyright (C) 2020 Freecss All rights reserved.
#
#   File Name     ：basic_model.py
#   Author        ：freecss
#   Email         ：karlfreecss@gmail.com
#   Created Date  ：2020/11/21
#   Description   ：
#
# ================================================================#

import sys

sys.path.append("..")

import torch
from torch.utils.data import Dataset

import os
from multiprocessing import Pool


class BasicDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        assert index < len(self), "index range error"

        img = self.X[index]
        label = self.Y[index]

        return (img, label)


class XYDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = torch.LongTensor(Y)

        self.n_sample = len(X)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        assert index < len(self), "index range error"

        img = self.X[index]
        if self.transform is not None:
            img = self.transform(img)

        label = self.Y[index]

        return (img, label)


class FakeRecorder:
    def __init__(self):
        pass

    def print(self, *x):
        pass


class BasicModel:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device,
        batch_size=1,
        num_epochs=1,
        stop_loss=0.01,
        num_workers=0,
        save_interval=None,
        save_dir=None,
        transform=None,
        collate_fn=None,
        recorder=None,
    ):

        self.model = model.to(device)

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.stop_loss = stop_loss
        self.num_workers = num_workers

        self.criterion = criterion
        self.optimizer = optimizer
        self.transform = transform
        self.device = device

        if recorder is None:
            recorder = FakeRecorder()
        self.recorder = recorder

        self.save_interval = save_interval
        self.save_dir = save_dir
        self.collate_fn = collate_fn
        pass

    def _fit(self, data_loader, n_epoch, stop_loss):
        recorder = self.recorder
        recorder.print("model fitting")

        min_loss = 1e10
        for epoch in range(n_epoch):
            loss_value = self.train_epoch(data_loader)
            recorder.print(f"{epoch}/{n_epoch} model training loss is {loss_value}")
            if min_loss < 0 or loss_value < min_loss:
                min_loss = loss_value
            if self.save_interval is not None and (epoch + 1) % self.save_interval == 0:
                assert self.save_dir is not None
                self.save(epoch + 1, self.save_dir)
            if stop_loss is not None and loss_value < stop_loss:
                break
        recorder.print("Model fitted, minimal loss is ", min_loss)
        return loss_value

    def fit(self, data_loader=None, X=None, y=None):
        if data_loader is None:
            data_loader = self._data_loader(X, y)
        return self._fit(data_loader, self.num_epochs, self.stop_loss)

    def train_epoch(self, data_loader):
        model = self.model
        criterion = self.criterion
        optimizer = self.optimizer
        device = self.device

        model.train()

        total_loss, total_num = 0.0, 0
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss = criterion(out, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.size(0)
            total_num += data.size(0)

        return total_loss / total_num

    def _predict(self, data_loader):
        model = self.model
        device = self.device

        model.eval()

        with torch.no_grad():
            results = []
            for data, _ in data_loader:
                data = data.to(device)
                out = model(data)
                results.append(out)

        return torch.cat(results, axis=0)

    def predict(self, data_loader=None, X=None, print_prefix=""):
        recorder = self.recorder
        recorder.print("Start Predict Class ", print_prefix)

        if data_loader is None:
            data_loader = self._data_loader(X)
        return self._predict(data_loader).argmax(axis=1).cpu().numpy()

    def predict_proba(self, data_loader=None, X=None, print_prefix=""):
        recorder = self.recorder
        # recorder.print('Start Predict Probability ', print_prefix)

        if data_loader is None:
            data_loader = self._data_loader(X)
        return self._predict(data_loader).softmax(axis=1).cpu().numpy()

    def _val(self, data_loader):
        model = self.model
        criterion = self.criterion
        device = self.device

        model.eval()

        total_correct_num, total_num, total_loss = 0, 0, 0.0

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)

                out = model(data)

                if len(out.shape) > 1:
                    correct_num = sum(target == out.argmax(axis=1)).item()
                else:
                    correct_num = sum(target == (out > 0.5)).item()
                loss = criterion(out, target)
                total_loss += loss.item() * data.size(0)

                total_correct_num += correct_num
                total_num += data.size(0)

        mean_loss = total_loss / total_num
        accuracy = total_correct_num / total_num

        return mean_loss, accuracy

    def val(self, data_loader=None, X=None, y=None, print_prefix=""):
        recorder = self.recorder
        recorder.print("Start val ", print_prefix)

        if data_loader is None:
            data_loader = self._data_loader(X, y)
        mean_loss, accuracy = self._val(data_loader)
        recorder.print(
            "[%s] Val loss: %f, accuray: %f" % (print_prefix, mean_loss, accuracy)
        )
        return accuracy

    def score(self, data_loader=None, X=None, y=None, print_prefix=""):
        return self.val(data_loader, X, y, print_prefix)

    def _data_loader(self, X, y=None):
        collate_fn = self.collate_fn
        transform = self.transform

        if y is None:
            y = [0] * len(X)
        dataset = XYDataset(X, y, transform=transform)
        sampler = None
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=int(self.num_workers),
            collate_fn=collate_fn,
        )
        return data_loader

    def save(self, epoch_id, save_dir):
        recorder = self.recorder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        recorder.print("Saving model and opter")
        save_path = os.path.join(save_dir, str(epoch_id) + "_net.pth")
        torch.save(self.model.state_dict(), save_path)

        save_path = os.path.join(save_dir, str(epoch_id) + "_opt.pth")
        torch.save(self.optimizer.state_dict(), save_path)

    def load(self, epoch_id, load_dir):
        recorder = self.recorder
        recorder.print("Loading model and opter")
        load_path = os.path.join(load_dir, str(epoch_id) + "_net.pth")
        self.model.load_state_dict(torch.load(load_path))

        load_path = os.path.join(load_dir, str(epoch_id) + "_opt.pth")
        self.optimizer.load_state_dict(torch.load(load_path))


if __name__ == "__main__":
    pass
