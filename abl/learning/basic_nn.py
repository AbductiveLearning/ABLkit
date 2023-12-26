from __future__ import annotations

import logging
import os
from typing import Any, Callable, List, Optional, Tuple

import numpy
import torch
from torch.utils.data import DataLoader

from ..utils.logger import print_log
from .torch_dataset import ClassificationDataset, PredictionDataset


class BasicNN:
    """
    Wrap NN models into the form of an sklearn estimator.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to be trained or used for prediction.
    loss_fn : torch.nn.Module
        The loss function used for training.
    optimizer : torch.optim.Optimizer
        The optimizer used for training.
    scheduler : Callable[..., Any], optional
        The learning rate scheduler used for training, which will be called
        at the end of each run of the ``fit`` method. It should implement the
        ``step`` method, by default None.
    device : torch.device, optional
        The device on which the model will be trained or used for prediction,
        by default torch.device("cpu").
    batch_size : int, optional
        The batch size used for training, by default 32.
    num_epochs : int, optional
        The number of epochs used for training, by default 1.
    stop_loss : float, optional
        The loss value at which to stop training, by default 0.0001.
    num_workers : int
        The number of workers used for loading data, by default 0.
    save_interval : int, optional
        The model will be saved every ``save_interval`` epochs during training, by default None.
    save_dir : str, optional
        The directory in which to save the model during training, by default None.
    train_transform : Callable[..., Any], optional
        A function/transform that takes an object and returns a transformed version used
        in the `fit` and `train_epoch` methods, by default None.
    test_transform : Callable[..., Any], optional
        A function/transform that takes an object and returns a transformed version in the
        `predict`, `predict_proba` and `score` methods, , by default None.
    collate_fn : Callable[[List[T]], Any], optional
        The function used to collate data, by default None.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Callable[..., Any]] = None,
        device: torch.device = torch.device("cpu"),
        batch_size: int = 32,
        num_epochs: int = 1,
        stop_loss: Optional[float] = 0.0001,
        num_workers: int = 0,
        save_interval: Optional[int] = None,
        save_dir: Optional[str] = None,
        train_transform: Callable[..., Any] = None,
        test_transform: Callable[..., Any] = None,
        collate_fn: Callable[[List[Any]], Any] = None,
    ) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be an instance of torch.nn.Module")
        if not isinstance(loss_fn, torch.nn.Module):
            raise TypeError("loss_fn must be an instance of torch.nn.Module")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be an instance of torch.optim.Optimizer")
        if scheduler is not None and not hasattr(scheduler, "step"):
            raise NotImplementedError("scheduler should implement the ``step`` method")
        if not isinstance(device, torch.device):
            raise TypeError("device must be an instance of torch.device")
        if not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if not isinstance(num_epochs, int):
            raise TypeError("num_epochs must be an integer")
        if stop_loss is not None and not isinstance(stop_loss, float):
            raise TypeError("stop_loss must be a float")
        if not isinstance(num_workers, int):
            raise TypeError("num_workers must be an integer")
        if save_interval is not None and not isinstance(save_interval, int):
            raise TypeError("save_interval must be an integer")
        if save_dir is not None and not isinstance(save_dir, str):
            raise TypeError("save_dir must be a string")
        if train_transform is not None and not callable(train_transform):
            raise TypeError("train_transform must be callable")
        if test_transform is not None and not callable(test_transform):
            raise TypeError("test_transform must be callable")
        if collate_fn is not None and not callable(collate_fn):
            raise TypeError("collate_fn must be callable")

        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.stop_loss = stop_loss
        self.num_workers = num_workers
        self.save_interval = save_interval
        self.save_dir = save_dir
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.collate_fn = collate_fn

        if self.save_interval is not None and self.save_dir is None:
            raise ValueError("save_dir should not be None if save_interval is not None.")

        if self.train_transform is not None and self.test_transform is None:
            print_log(
                "Transform used in the training phase will be used in prediction.",
                logger="current",
                level=logging.WARNING,
            )
            self.test_transform = self.train_transform

    def _fit(self, data_loader: DataLoader) -> BasicNN:
        """
        Internal method to fit the model on data for ``self.num_epochs`` times,
        with early stopping.

        Parameters
        ----------
        data_loader : DataLoader
            Data loader providing training samples.

        Returns
        -------
        BasicNN
            The model itself after training.
        """
        if not isinstance(data_loader, DataLoader):
            raise TypeError(
                f"data_loader must be an instance of torch.utils.data.DataLoader, "
                f"but got {type(data_loader)}"
            )

        for epoch in range(self.num_epochs):
            loss_value = self.train_epoch(data_loader)
            if self.save_interval is not None and (epoch + 1) % self.save_interval == 0:
                self.save(epoch + 1)
            if self.stop_loss is not None and loss_value < self.stop_loss:
                break
        if self.scheduler is not None:
            self.scheduler.step()
        print_log(f"model loss: {loss_value:.5f}", logger="current")
        return self

    def fit(
        self, data_loader: DataLoader = None, X: List[Any] = None, y: List[int] = None
    ) -> BasicNN:
        """
        Train the model for self.num_epochs times or until the average loss on one epoch
        is less than self.stop_loss. It supports training with either a DataLoader
        object (data_loader) or a pair of input data (X) and target labels (y). If both
        data_loader and (X, y) are provided, the method will prioritize using the data_loader.

        Parameters
        ----------
        data_loader : DataLoader, optional
            The data loader used for training, by default None.
        X : List[Any], optional
            The input data, by default None.
        y : List[int], optional
            The target data, by default None.

        Returns
        -------
        BasicNN
            The model itself after training.
        """
        if data_loader is not None and X is not None:
            print_log(
                "data_loader will be used to train the model instead of X and y.",
                logger="current",
                level=logging.WARNING,
            )
        if data_loader is None:
            if X is None:
                raise ValueError("data_loader and X can not be None simultaneously.")
            else:
                data_loader = self._data_loader(X, y)
        return self._fit(data_loader)

    def train_epoch(self, data_loader: DataLoader) -> float:
        """
        Train the model with an instance of DataLoader (data_loader) for one epoch.

        Parameters
        ----------
        data_loader : DataLoader
            The data loader used for training.

        Returns
        -------
        float
            The average loss on one epoch.
        """
        model = self.model
        loss_fn = self.loss_fn
        optimizer = self.optimizer
        device = self.device

        model.train()

        total_loss, total_num = 0.0, 0
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss = loss_fn(out, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.size(0)
            total_num += data.size(0)

        return total_loss / total_num

    def _predict(self, data_loader: DataLoader) -> torch.Tensor:
        """
        Internal method to predict the outputs given a DataLoader.

        Parameters
        ----------
        data_loader : DataLoader
            The DataLoader providing input samples.

        Returns
        -------
        torch.Tensor
            Raw output from the model.
        """
        if not isinstance(data_loader, DataLoader):
            raise TypeError(
                f"data_loader must be an instance of torch.utils.data.DataLoader, "
                f"but got {type(data_loader)}"
            )
        model = self.model
        device = self.device

        model.eval()

        with torch.no_grad():
            results = []
            for data in data_loader:
                data = data.to(device)
                out = model(data)
                results.append(out)

        return torch.cat(results, axis=0)

    def predict(self, data_loader: DataLoader = None, X: List[Any] = None) -> numpy.ndarray:
        """
        Predict the class of the input data. This method supports prediction with either
        a DataLoader object (data_loader) or a list of input data (X). If both data_loader
        and X are provided, the method will predict the input data in data_loader
        instead of X.

        Parameters
        ----------
        data_loader : DataLoader, optional
            The data loader used for prediction, by default None.
        X : List[Any], optional
            The input data, by default None.

        Returns
        -------
        numpy.ndarray
            The predicted class of the input data.
        """

        if data_loader is not None and X is not None:
            print_log(
                "Predict the class of input data in data_loader instead of X.",
                logger="current",
                level=logging.WARNING,
            )

        if data_loader is None:
            dataset = PredictionDataset(X, self.test_transform)
            data_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=int(self.num_workers),
                collate_fn=self.collate_fn,
            )
        return self._predict(data_loader).argmax(axis=1).cpu().numpy()

    def predict_proba(self, data_loader: DataLoader = None, X: List[Any] = None) -> numpy.ndarray:
        """
        Predict the probability of each class for the input data. This method supports
        prediction with either a DataLoader object (data_loader) or a list of input data (X).
        If both data_loader and X are provided, the method will predict the input data in
        data_loader instead of X.

        Parameters
        ----------
        data_loader : DataLoader, optional
            The data loader used for prediction, by default None.
        X : List[Any], optional
            The input data, by default None.

        Returns
        -------
        numpy.ndarray
            The predicted probability of each class for the input data.
        """

        if data_loader is not None and X is not None:
            print_log(
                "Predict the class probability of input data in data_loader instead of X.",
                logger="current",
                level=logging.WARNING,
            )

        if data_loader is None:
            dataset = PredictionDataset(X, self.test_transform)
            data_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=int(self.num_workers),
                collate_fn=self.collate_fn,
            )
        return self._predict(data_loader).softmax(axis=1).cpu().numpy()

    def _score(self, data_loader: DataLoader) -> Tuple[float, float]:
        """
        Internal method to compute loss and accuracy for the data provided through a DataLoader.

        Parameters
        ----------
        data_loader : DataLoader
            Data loader to use for evaluation.

        Returns
        -------
        Tuple[float, float]
            mean_loss: float, The mean loss of the model on the provided data.
            accuracy: float, The accuracy of the model on the provided data.
        """
        if not isinstance(data_loader, DataLoader):
            raise TypeError(
                f"data_loader must be an instance of torch.utils.data.DataLoader, "
                f"but got {type(data_loader)}"
            )

        model = self.model
        loss_fn = self.loss_fn
        device = self.device

        model.eval()

        total_correct_num, total_num, total_loss = 0, 0, 0.0

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)

                out = model(data)

                if len(out.shape) > 1:
                    correct_num = (target == out.argmax(axis=1)).sum().item()
                else:
                    correct_num = (target == (out > 0.5)).sum().item()
                loss = loss_fn(out, target)
                total_loss += loss.item() * data.size(0)

                total_correct_num += correct_num
                total_num += data.size(0)

        mean_loss = total_loss / total_num
        accuracy = total_correct_num / total_num

        return mean_loss, accuracy

    def score(
        self, data_loader: DataLoader = None, X: List[Any] = None, y: List[int] = None
    ) -> float:
        """
        Validate the model. It supports validation with either a DataLoader object (data_loader)
        or a pair of input data (X) and ground truth labels (y). If both data_loader and
        (X, y) are provided, the method will prioritize using the data_loader.

        Parameters
        ----------
        data_loader : DataLoader, optional
            The data loader used for scoring, by default None.
        X : List[Any], optional
            The input data, by default None.
        y : List[int], optional
            The target data, by default None.

        Returns
        -------
        float
            The accuracy of the model.
        """
        print_log("Start machine learning model validation", logger="current")

        if data_loader is not None and X is not None:
            print_log(
                "data_loader will be used to validate the model instead of X and y.",
                logger="current",
                level=logging.WARNING,
            )

        if data_loader is None:
            if X is None or y is None:
                raise ValueError("data_loader and (X, y) can not be None simultaneously.")
            else:
                data_loader = self._data_loader(X, y)
        mean_loss, accuracy = self._score(data_loader)
        print_log(f"mean loss: {mean_loss:.3f}, accuray: {accuracy:.3f}", logger="current")
        return accuracy

    def _data_loader(self, X: List[Any], y: List[int] = None, shuffle: bool = True) -> DataLoader:
        """
        Generate a DataLoader for user-provided input data and target labels.

        Parameters
        ----------
        X : List[Any]
            Input samples.
        y : List[int], optional
            Target labels. If None, dummy labels are created, by default None.
        shuffle : bool, optional
            Whether to shuffle the data, by default True.

        Returns
        -------
        DataLoader
            A DataLoader providing batches of (X, y) pairs.
        """

        if X is None:
            raise ValueError("X should not be None.")
        if y is None:
            y = [0] * len(X)
        if not (len(y) == len(X)):
            raise ValueError("X and y should have equal length.")

        dataset = ClassificationDataset(X, y, transform=self.train_transform)
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=int(self.num_workers),
            collate_fn=self.collate_fn,
        )
        return data_loader

    def save(self, epoch_id: int = 0, save_path: str = None) -> None:
        """
        Save the model and the optimizer. User can either provide a save_path or specify
        the epoch_id at which the model and optimizer is saved. if both save_path and
        epoch_id are provided, save_path will be used. If only epoch_id is specified,
        model and optimizer will be saved to the path f"model_checkpoint_epoch_{epoch_id}.pth"
        under ``self.save_dir``. save_path and epoch_id can not be None simultaneously.

        Parameters
        ----------
        epoch_id : int
            The epoch id.
        save_path : str, optional
            The path to save the model, by default None.
        """
        if self.save_dir is None and save_path is None:
            raise ValueError("'save_dir' and 'save_path' should not be None simultaneously.")

        if save_path is not None:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
        else:
            save_path = os.path.join(self.save_dir, f"model_checkpoint_epoch_{epoch_id}.pth")
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        print_log(f"Checkpoints will be saved to {save_path}", logger="current")

        save_parma_dic = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        torch.save(save_parma_dic, save_path)

    def load(self, load_path: str) -> None:
        """
        Load the model and the optimizer.

        Parameters
        ----------
        load_path : str
            The directory to load the model, by default "".
        """

        if load_path is None:
            raise ValueError("Load path should not be None.")

        print_log(
            f"Loads checkpoint by local backend from path: {load_path}",
            logger="current",
        )

        param_dic = torch.load(load_path)
        self.model.load_state_dict(param_dic["model"])
        if "optimizer" in param_dic.keys():
            self.optimizer.load_state_dict(param_dic["optimizer"])
