from typing import Any, Callable, List, Tuple

import torch
from torch.utils.data import Dataset


class PredictionDataset(Dataset):
    """
    Dataset used for prediction.

    Parameters
    ----------
    X : List[Any]
        The input data.
    transform : Callable[..., Any], optional
        A function/transform that takes an object and returns a transformed version.
        Defaults to None.
    """

    def __init__(self, X: List[Any], transform: Callable[..., Any] = None):
        if not isinstance(X, list):
            raise ValueError("X should be of type list.")

        self.X = X
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
        if index >= len(self):
            raise ValueError("index range error")

        x = self.X[index]
        if self.transform is not None:
            x = self.transform(x)
        return x
