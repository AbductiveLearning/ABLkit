"""
Implementation of PyTorch dataset class used for multi-label classification.

Copyright (c) 2024 LAMDA.  All rights reserved.
"""

from typing import Any, Callable, List, Optional

import numpy as np
import torch

from .classification_dataset import ClassificationDataset


class MultiLabelClassificationDataset(ClassificationDataset):
    """
    Dataset used for multi-label classification, where each target ``Y[i]``
    is a binary indicator vector (one entry per label) rather than a single
    class index. ``Y`` is stored as a ``float32`` tensor so it can be fed
    directly into ``BCEWithLogitsLoss`` and similar losses.

    Parameters
    ----------
    X : List[Any]
        The input data.
    Y : List[Any]
        The per-sample label vectors. Each entry is converted via
        ``numpy.stack`` and stored as a ``FloatTensor``.
    transform : Callable[..., Any], optional
        A function/transform that takes an object and returns a transformed
        version. Defaults to None.
    """

    def __init__(
        self,
        X: List[Any],
        Y: List[Any],
        transform: Optional[Callable[..., Any]] = None,
    ) -> None:
        if (not isinstance(X, list)) or (not isinstance(Y, list)):
            raise ValueError("X and Y should be of type list.")
        if len(X) != len(Y):
            raise ValueError("Length of X and Y must be equal.")

        self.X = X
        self.Y = torch.FloatTensor(np.stack(Y, axis=0))
        self.transform = transform
