import torch
from torch.utils.data import Dataset
from typing import List, Any, Tuple


class RegressionDataset(Dataset):
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

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Get an item from the dataset.

        Parameters
        ----------
        index : int
            The index of the item to retrieve.

        Returns
        -------
        Tuple[Any, Any]
            A tuple containing the input and output data at the specified index.
        """
        if index >= len(self):
            raise ValueError("index range error")

        x = self.X[index]
        y = self.Y[index]

        return x, y
