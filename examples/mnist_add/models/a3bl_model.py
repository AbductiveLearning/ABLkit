import logging
from typing import Any, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ablkit.learning.basic_nn import BasicNN
from ablkit.learning.torch_dataset import PredictionDataset
from ablkit.utils import print_log


class A3BLBasicNN(BasicNN):
    """
    A ``BasicNN`` variant that exposes ``extract_features`` so the wrapping
    ``ABLModel`` can populate ``data_example.embeddings`` for the A3BL
    pipeline and any ``dist_func='similarity'`` consumers.

    The underlying PyTorch ``model`` must implement ``extract_features(x)``
    returning the representation used for similarity computations.
    """

    def _extract_features(self, data_loader: DataLoader) -> torch.Tensor:
        """
        Run the model's ``extract_features`` over every batch in
        ``data_loader`` and concatenate the results.

        Parameters
        ----------
        data_loader : DataLoader
            The DataLoader providing input samples.

        Returns
        -------
        torch.Tensor
            Concatenated feature tensor across all batches.
        """
        if not isinstance(data_loader, DataLoader):
            raise TypeError(
                "data_loader must be an instance of torch.utils.data.DataLoader, "
                f"but got {type(data_loader)}"
            )
        model = self.model
        device = self.device

        model.eval()
        with torch.no_grad():
            results = []
            for data in data_loader:
                data = data.to(device)
                results.append(model.extract_features(data))

        return torch.cat(results, dim=0)

    def extract_features(
        self,
        data_loader: Optional[DataLoader] = None,
        X: Optional[List[Any]] = None,
    ) -> np.ndarray:
        """
        Compute feature embeddings for ``X`` (or a prebuilt ``data_loader``).
        When both are provided, ``data_loader`` takes precedence.

        Parameters
        ----------
        data_loader : DataLoader, optional
            DataLoader to use directly. Defaults to None.
        X : List[Any], optional
            Raw input list; converted to a ``PredictionDataset`` when used.
            Defaults to None.

        Returns
        -------
        np.ndarray
            Feature embeddings as a NumPy array of shape
            ``(num_samples, embedding_dim)``.
        """
        if data_loader is not None and X is not None:
            print_log(
                "Extracting features from data_loader; ignoring X.",
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
        return self._extract_features(data_loader).cpu().numpy()
