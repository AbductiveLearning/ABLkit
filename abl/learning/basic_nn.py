import logging
import os
from typing import Any, Callable, List, Optional, T, Tuple

import numpy
import torch
from torch.utils.data import DataLoader

from ..dataset import ClassificationDataset, PredictionDataset
from ..utils.logger import print_log


class BasicNN:
    """
    Wrap NN models into the form of an sklearn estimator.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to be trained or used for prediction.
    criterion : torch.nn.Module
        The loss function used for training.
    optimizer : torch.optim.Optimizer
        The optimizer used for training.
    device : torch.device, optional
        The device on which the model will be trained or used for prediction,
        by default torch.device("cpu").
    batch_size : int, optional
        The batch size used for training, by default 32.
    num_epochs : int, optional
        The number of epochs used for training, by default 1.
    stop_loss : Optional[float], optional
        The loss value at which to stop training, by default 0.01.
    num_workers : int
        The number of workers used for loading data, by default 0.
    save_interval : Optional[int], optional
        The interval at which to save the model during training, by default None.
    save_dir : Optional[str], optional
        The directory in which to save the model during training, by default None.
    train_transform : Callable[..., Any], optional
        A function/transform that takes in an object and returns a transformed version used
        in the `fit` and `train_epoch` methods, by default None.
    test_transform : Callable[..., Any], optional
        A function/transform that takes in an object and returns a transformed version in the
        `predict`, `predict_proba` and `score` methods, , by default None.
    collate_fn : Callable[[List[T]], Any], optional
        The function used to collate data, by default None.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device = torch.device("cpu"),
        batch_size: int = 32,
        num_epochs: int = 1,
        stop_loss: Optional[float] = 0.01,
        num_workers: int = 0,
        save_interval: Optional[int] = None,
        save_dir: Optional[str] = None,
        train_transform: Callable[..., Any] = None,
        test_transform: Callable[..., Any] = None,
        collate_fn: Callable[[List[T]], Any] = None,
    ) -> None:
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
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

        if self.train_transform is not None and self.test_transform is None:
            print_log(
                "Transform used in the training phase will be used in prediction.",
                logger="current",
                level=logging.WARNING,
            )
            self.test_transform = self.train_transform

    def _fit(self, data_loader: DataLoader) -> float:
        """
        Internal method to fit the model on data for n epochs, with early stopping.

        Parameters
        ----------
        data_loader : DataLoader
            Data loader providing training samples.

        Returns
        -------
        float
            The loss value of the trained model.
        """
        for epoch in range(self.num_epochs):
            loss_value = self.train_epoch(data_loader)
            if self.save_interval is not None and (epoch + 1) % self.save_interval == 0:
                if self.save_dir is None:
                    raise ValueError("save_dir should not be None if save_interval is not None.")
                self.save(epoch + 1)
            if self.stop_loss is not None and loss_value < self.stop_loss:
                break
        print_log(f"model loss: {loss_value:.5f}", logger="current")
        return self

    def fit(
        self, data_loader: DataLoader = None, X: List[Any] = None, y: List[int] = None
    ) -> float:
        """
        Train the model.

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
        float
            The loss value of the trained model.
        """
        if data_loader is None:
            if X is None:
                raise ValueError("data_loader and X can not be None simultaneously.")
            else:
                data_loader = self._data_loader(X, y)
        return self._fit(data_loader)

    def train_epoch(self, data_loader: DataLoader) -> float:
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
        Predict the class of the input data.

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
        Predict the probability of each class for the input data.

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
                    correct_num = (target == out.argmax(axis=1)).sum().item()
                else:
                    correct_num = (target == (out > 0.5)).sum().item()
                loss = criterion(out, target)
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
        Validate the model.

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

        if data_loader is None:
            data_loader = self._data_loader(X, y)
        mean_loss, accuracy = self._score(data_loader)
        print_log(f"mean loss: {mean_loss:.3f}, accuray: {accuracy:.3f}", logger="current")
        return accuracy

    def _data_loader(self, X: List[Any], y: List[int] = None, shuffle: bool = True) -> DataLoader:
        """
        Generate a DataLoader for user-provided input and target data.

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
        Save the model and the optimizer.

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

    def load(self, load_path: str = "") -> None:
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
