from typing import Any, List, Optional, Tuple

from torch.utils.data import Dataset


class BridgeDataset(Dataset):
    """Dataset used in ``BaseBridge``.

    Parameters
    ----------
    X : List[List[Any]]
        A list of objects representing the input data.
    gt_pseudo_label : List[List[Any]], optional
        A list of objects representing the ground truth label of each element in ``X``.
    Y : List[Any]
        A list of objects representing the ground truth of the reasoning result of
        each instance in ``X``.
    """

    def __init__(
        self,
        X: List[List[Any]],
        gt_pseudo_label: Optional[List[List[Any]]],
        Y: List[Any],
    ):
        if (not isinstance(X, list)) or (not isinstance(Y, list)):
            raise ValueError("X and Y should be of type list.")
        if len(X) != len(Y):
            raise ValueError("Length of X and Y must be equal.")

        self.X = X
        self.gt_pseudo_label = gt_pseudo_label
        self.Y = Y

        if self.gt_pseudo_label is None:
            self.gt_pseudo_label = [None] * len(self.X)

    def __len__(self):
        """Return the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.
        """
        return len(self.X)

    def __getitem__(self, index: int) -> Tuple[List, List, Any]:
        """Get an item from the dataset.

        Parameters
        ----------
        index : int
            The index of the item to retrieve.

        Returns
        -------
        Tuple[List, List, Any]
            A tuple containing the input and output data at the specified index.
        """
        if index >= len(self):
            raise ValueError("index range error")

        X = self.X[index]
        gt_pseudo_label = self.gt_pseudo_label[index]
        Y = self.Y[index]

        return (X, gt_pseudo_label, Y)
