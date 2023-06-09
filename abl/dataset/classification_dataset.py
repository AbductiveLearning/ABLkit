import torch
from torch.utils.data import Dataset
from typing import List, Any, Tuple, Callable


class ClassificationDataset(Dataset):
    def __init__(
        self, X: List[Any], Y: List[int], transform: Callable[..., Any] = None
    ):
        """
        Initialize the dataset used for classification task.

        Parameters
        ----------
        X : List[Any]
            The input data.
        Y : List[int]
            The target data.
        transform : Callable[..., Any], optional
            A function/transform that takes in an object and returns a transformed version. Defaults to None.
        """
        self.X = X
        self.Y = torch.LongTensor(Y)
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

        y = self.Y[index]

        return x, y