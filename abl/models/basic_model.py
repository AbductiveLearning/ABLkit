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
import numpy
from torch.utils.data import Dataset, DataLoader

import os
from multiprocessing import Pool
from typing import List, Any, T, Tuple, Optional, Callable


class BasicDataset(Dataset):
    def __init__(self, X: List[Any], Y: List[Any]):
        """Initialize a basic dataset.

        Parameters
        ----------
        X : List[Any]
            A list of objects representing the input data.
        Y : List[Any]
            A list of objects representing the output data.
        """
        self.X = X
        self.Y = Y

    def __len__(self):
        """Return the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.
        """
        return len(self.X)

    def __getitem__(self, index: int) -> Tuple(Any, Any):
        """Get an item from the dataset.

        Parameters
        ----------
        index : int
            The index of the item to retrieve.

        Returns
        -------
        Tuple(Any, Any)
            A tuple containing the input and output data at the specified index.
        """
        assert index < len(self), "index range error"

        img = self.X[index]
        label = self.Y[index]

        return (img, label)


class XYDataset(Dataset):
    def __init__(self, X: List[Any], Y: List[int], transform: Callable[...] = None):
        """
        Initialize the dataset used for classification task.

        Parameters
        ----------
        X : List[Any]
            The input data.
        Y : List[int]
            The target data.
        transform : callable, optional
            A function/transform that takes in an object and returns a transformed version. Defaults to None.
        """
        self.X = X
        self.Y = torch.LongTensor(Y)

        self.n_sample = len(X)
        self.transform = transform

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.
        """
        return len(self.X)

    def __getitem__(self, index: int) -> Tuple[Any, torch.Tensor]:
        """
        Get the item at the given index.

        Parameters
        ----------
        index : int
            The index of the item to get.

        Returns
        -------
        Tuple[Any, torch.Tensor]
            A tuple containing the object and its label.
        """
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
    """
    Wrap NN models into the form of an sklearn estimator

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to be trained or used for prediction.
    criterion : torch.nn.Module
        The loss function used for training.
    optimizer : torch.nn.Module
        The optimizer used for training.
    device : torch.device
        The device on which the model will be trained or used for prediction.
    batch_size : int, optional
        The batch size used for training, by default 1.
    num_epochs : int, optional
        The number of epochs used for training, by default 1.
    stop_loss : Optional[float], optional
        The loss value at which to stop training, by default 0.01.
    num_workers : int, optional
        The number of workers used for loading data, by default 0.
    save_interval : Optional[int], optional
        The interval at which to save the model during training, by default None.
    save_dir : Optional[str], optional
        The directory in which to save the model during training, by default None.
    transform : Callable[..., Any], optional
        The transformation function used for data augmentation, by default None.
    collate_fn : Callable[[List[T]], Any], optional
        The function used to collate data, by default None.
    recorder : Any, optional
        The recorder used to record training progress, by default None.

    Attributes
    ----------
    model : torch.nn.Module
        The PyTorch model to be trained or used for prediction.
    batch_size : int
        The batch size used for training.
    num_epochs : int
        The number of epochs used for training.
    stop_loss : Optional[float]
        The loss value at which to stop training.
    num_workers : int
        The number of workers used for loading data.
    criterion : torch.nn.Module
        The loss function used for training.
    optimizer : torch.nn.Module
        The optimizer used for training.
    transform : Callable[..., Any]
        The transformation function used for data augmentation.
    device : torch.device
        The device on which the model will be trained or used for prediction.
    recorder : Any
        The recorder used to record training progress.
    save_interval : Optional[int]
        The interval at which to save the model during training.
    save_dir : Optional[str]
        The directory in which to save the model during training.
    collate_fn : Callable[[List[T]], Any]
        The function used to collate data.

    Methods
    -------
    fit(data_loader=None, X=None, y=None)
        Train the model.
    train_epoch(data_loader)
        Train the model for one epoch.
    predict(data_loader=None, X=None, print_prefix="")
        Predict the class of the input data.
    predict_proba(data_loader=None, X=None, print_prefix="")
        Predict the probability of each class for the input data.
    val(data_loader=None, X=None, y=None, print_prefix="")
        Validate the model.
    score(data_loader=None, X=None, y=None, print_prefix="")
        Score the model.
    _data_loader(X, y=None)
        Load data.
    save(epoch_id, save_dir="")
        Save the model.
    load(epoch_id, load_dir="")
        Load the model.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.nn.Module,
        device: torch.device,
        batch_size: int = 1,
        num_epochs: int = 1,
        stop_loss: Optional[float] = 0.01,
        num_workers: int = 0,
        save_interval: Optional[int] = None,
        save_dir: Optional[str] = None,
        transform: Callable[...] = None,
        collate_fn: Callable[[List[T]], Any] = None,
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

    def fit(
        self, data_loader: DataLoader = None, X: List[Any] = None, y: List[int] = None
    ) -> float:
        """
        Train the model.

        Parameters
        ----------
        data_loader : DataLoader, optional
            The data loader used for training, by default None
        X : List[Any], optional
            The input data, by default None
        y : List[int], optional
            The target data, by default None

        Returns
        -------
        float
            The loss value of the trained model.
        """
        if data_loader is None:
            data_loader = self._data_loader(X, y)
        return self._fit(data_loader, self.num_epochs, self.stop_loss)

    def train_epoch(self, data_loader: DataLoader):
        """
        Train the model for one epoch.

        Parameters
        ----------
        data_loader : DataLoader
            The data loader used for training.

        Returns
        -------
        float
            The loss value of the trained model.
        """
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

    def predict(
        self,
        data_loader: DataLoader = None,
        X: List[Any] = None,
        print_prefix: str = "",
    ) -> numpy.ndarray:
        """
        Predict the class of the input data.

        Parameters
        ----------
        data_loader : DataLoader, optional
            The data loader used for prediction, by default None
        X : List[Any], optional
            The input data, by default None
        print_prefix : str, optional
            The prefix used for printing, by default ""

        Returns
        -------
        numpy.ndarray
            The predicted class of the input data.
        """
        recorder = self.recorder
        recorder.print("Start Predict Class ", print_prefix)

        if data_loader is None:
            data_loader = self._data_loader(X)
        return self._predict(data_loader).argmax(axis=1).cpu().numpy()

    def predict_proba(
        self,
        data_loader: DataLoader = None,
        X: List[Any] = None,
        print_prefix: str = "",
    ) -> numpy.ndarray:
        """
        Predict the probability of each class for the input data.

        Parameters
        ----------
        data_loader : DataLoader, optional
            The data loader used for prediction, by default None
        X : List[Any], optional
            The input data, by default None
        print_prefix : str, optional
            The prefix used for printing, by default ""

        Returns
        -------
        numpy.ndarray
            The predicted probability of each class for the input data.
        """
        recorder = self.recorder
        recorder.print("Start Predict Probability ", print_prefix)

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

    def val(
        self,
        data_loader: DataLoader = None,
        X: List[Any] = None,
        y: List[int] = None,
        print_prefix: str = "",
    ) -> float:
        """
        Validate the model.

        Parameters
        ----------
        data_loader : DataLoader, optional
            The data loader used for validation, by default None
        X : List[Any], optional
            The input data, by default None
        y : List[int], optional
            The target data, by default None
        print_prefix : str, optional
            The prefix used for printing, by default ""

        Returns
        -------
        float
            The accuracy of the model.
        """
        recorder = self.recorder
        recorder.print("Start val ", print_prefix)

        if data_loader is None:
            data_loader = self._data_loader(X, y)
        mean_loss, accuracy = self._val(data_loader)
        recorder.print(
            "[%s] Val loss: %f, accuray: %f" % (print_prefix, mean_loss, accuracy)
        )
        return accuracy

    def score(
        self,
        data_loader: DataLoader = None,
        X: List[Any] = None,
        y: List[int] = None,
        print_prefix: str = "",
    ) -> float:
        """
        Score the model.

        Parameters
        ----------
        data_loader : DataLoader, optional
            The data loader used for scoring, by default None
        X : List[Any], optional
            The input data, by default None
        y : List[int], optional
            The target data, by default None
        print_prefix : str, optional
            The prefix used for printing, by default ""

        Returns
        -------
        float
            The accuracy of the model.
        """
        return self.val(data_loader, X, y, print_prefix)

    def _data_loader(
        self,
        X: List[Any],
        y: List[int] = None,
    ) -> DataLoader:
        """
        Generate data_loader for user provided data.

        Parameters
        ----------
        X : List[Any]
            The input data.
        y : List[int], optional
            The target data, by default None

        Returns
        -------
        DataLoader
            The data loader.
        """
        collate_fn = self.collate_fn
        transform = self.transform

        if y is None:
            y = [0] * len(X)
        dataset = XYDataset(X, y, transform=transform)
        sampler = None
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=int(self.num_workers),
            collate_fn=collate_fn,
        )
        return data_loader

    def save(self, epoch_id: int, save_dir: str = ""):
        """
        Save the model and the optimizer.

        Parameters
        ----------
        epoch_id : int
            The epoch id.
        save_dir : str, optional
            The directory to save the model, by default ""
        """
        recorder = self.recorder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        recorder.print("Saving model and opter")
        save_path = os.path.join(save_dir, str(epoch_id) + "_net.pth")
        torch.save(self.model.state_dict(), save_path)

        save_path = os.path.join(save_dir, str(epoch_id) + "_opt.pth")
        torch.save(self.optimizer.state_dict(), save_path)

    def load(self, epoch_id: int, load_dir: str = ""):
        """
        Load the model and the optimizer.

        Parameters
        ----------
        epoch_id : int
            The epoch id.
        load_dir : str, optional
            The directory to load the model, by default ""
        """
        recorder = self.recorder
        recorder.print("Loading model and opter")
        load_path = os.path.join(load_dir, str(epoch_id) + "_net.pth")
        self.model.load_state_dict(torch.load(load_path))

        load_path = os.path.join(load_dir, str(epoch_id) + "_opt.pth")
        self.optimizer.load_state_dict(torch.load(load_path))


if __name__ == "__main__":
    pass
