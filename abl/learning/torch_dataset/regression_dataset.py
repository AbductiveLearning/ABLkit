from typing import Any, List, Tuple

from torch.utils.data import Dataset


class RegressionDataset(Dataset):
    """
    Dataset used for regression task.

    Parameters
    ----------
    X : List[Any]
        A list of objects representing the input data.
    Y : List[Any]
        A list of objects representing the output data.
    """

    def __init__(self, X: List[Any], Y: List[Any]):
        if (not isinstance(X, list)) or (not isinstance(Y, list)):
            raise ValueError("X and Y should be of type list.")
        if len(X) != len(Y):
            raise ValueError("Length of X and Y must be equal.")

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
