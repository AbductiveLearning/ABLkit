import logging
import os
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy
import torch
from torch.utils.data import DataLoader, Dataset

from ablkit.learning import BasicNN, PredictionDataset, ClassificationDataset
from ablkit.utils.logger import print_log


class MultiLabelClassificationDataset(ClassificationDataset):
    def __init__(self, X: List[Any], Y: List[int], transform: Optional[Callable[..., Any]] = None):
        if (not isinstance(X, list)) or (not isinstance(Y, list)):
            raise ValueError("X and Y should be of type list.")
        self.X = X
        self.Y = torch.FloatTensor(numpy.stack(Y, axis=0)) # float32 for BCELoss
        self.transform = transform

class BDDNN(BasicNN):

    def predict(
        self,
        data_loader: Optional[DataLoader] = None,
        X: Optional[List[Any]] = None,
    ) -> numpy.ndarray:
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
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                pin_memory=torch.cuda.is_available(),
            )
        pred_probs = self._predict(data_loader).sigmoid()
        pred = torch.where(pred_probs > 0.5, 1, 0).int()
        return pred.cpu().numpy()

    def predict_proba(
        self,
        data_loader: Optional[DataLoader] = None,
        X: Optional[List[Any]] = None,
    ) -> numpy.ndarray:
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
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                pin_memory=torch.cuda.is_available(),
            )
        pred_probs = self._predict(data_loader).sigmoid()  # B x NC
        return pred_probs.cpu().numpy()

    def _data_loader(
        self,
        X: Optional[List[Any]],
        y: Optional[List[int]] = None,
        shuffle: Optional[bool] = True,
    ) -> DataLoader:
        if X is None:
            raise ValueError("X should not be None.")
        if y is None:
            y = [0] * len(X)
        if not len(y) == len(X):
            raise ValueError("X and y should have equal length.")

        dataset = MultiLabelClassificationDataset(X, y, transform=self.train_transform)
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=torch.cuda.is_available(),
        )
        return data_loader
