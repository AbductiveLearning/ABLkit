from torch.utils.data import Dataset
from typing import List, Any, Tuple


class BridgeDataset(Dataset):
    def __init__(self, X: List[Any], Z: List[Any], Y: List[Any]):
        """Initialize a basic dataset.

        Parameters
        ----------
        X : List[Any]
            A list of objects representing the input data.
        Z : List[Any]
            A list of objects representing the symbol.
        Y : List[Any]
            A list of objects representing the label.
        """
        if (not isinstance(X, list)) or (not isinstance(Y, list)):
            raise ValueError("X and Y should be of type list.")
        if len(X) != len(Y):
            raise ValueError("Length of X and Y must be equal.")
        
        self.X = X
        self.Z = Z
        self.Y = Y

        if self.Z is None:
            self.Z = [None] * len(self.X)

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

        X = self.X[index]
        Z = self.Z[index]
        Y = self.Y[index]

        return (X, Z, Y)