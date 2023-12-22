# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import numpy as np
import torch

from ...utils import flatten as flatten_list
from ...utils import to_hashable
from .base_data_element import BaseDataElement

BoolTypeTensor = Union[torch.BoolTensor, torch.cuda.BoolTensor]
LongTypeTensor = Union[torch.LongTensor, torch.cuda.LongTensor]

IndexType = Union[str, slice, int, list, LongTypeTensor, BoolTypeTensor, np.ndarray]


# Modified from
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/data_structures/instance_data.py # noqa
class ListData(BaseDataElement):
    """
    Abstract Data Interface used throughout the ABL-Package.

    ``ListData`` is the underlying data structure used in the ABL-Package,
    designed to manage diverse forms of data dynamically generated throughout the
    Abductive Learning (ABL) framework. This includes handling raw data, predicted
    pseudo-labels, abduced pseudo-labels, pseudo-label indices, etc.

    As a fundamental data structure in ABL, ``ListData`` is essential for the smooth
    transfer and manipulation of data across various components of the ABL framework,
    such as prediction, abductive reasoning, and training phases. It provides a
    unified data format across these stages, ensuring compatibility and flexibility
    in handling diverse data forms in the ABL framework.

    The attributes in ``ListData`` are divided into two parts,
    the ``metainfo`` and the ``data`` respectively.

        - ``metainfo``: Usually used to store basic information about data examples,
          such as symbol number, image size, etc. The attributes can be accessed or
          modified by dict-like or object-like operations, such as ``.`` (for data
          access and modification), ``in``, ``del``, ``pop(str)``, ``get(str)``,
          ``metainfo_keys()``, ``metainfo_values()``, ``metainfo_items()``,
          ``set_metainfo()`` (for set or change key-value pairs in metainfo).

        - ``data``: raw data, labels, predictions, and abduced results are stored.
          The attributes can be accessed or modified by dict-like or object-like operations,
          such as ``.``, ``in``, ``del``, ``pop(str)``, ``get(str)``, ``keys()``,
          ``values()``, ``items()``. Users can also apply tensor-like
          methods to all :obj:`torch.Tensor` in the ``data_fields``, such as ``.cuda()``,
          ``.cpu()``, ``.numpy()``, ``.to()``, ``to_tensor()``, ``.detach()``.

    ListData supports ``index`` and ``slice`` for data field. The type of value in
    data field can be either ``None`` or ``list`` of base data structures such as
    ``torch.Tensor``, ``numpy.ndarray``, ``list``, ``str`` and ``tuple``.

    This design is inspired by and extends the functionalities of the ``BaseDataElement``
    class implemented in `MMEngine <https://github.com/open-mmlab/mmengine/blob/main/mmengine/structures/base_data_element.py>`_.

    Examples:
        >>> from abl.data.structures import ListData
        >>> import numpy as np
        >>> import torch
        >>> data_examples = ListData()
        >>> data_examples.X = [list(torch.randn(2)) for _ in range(3)]
        >>> data_examples.Y = [1, 2, 3]
        >>> data_examples.gt_pseudo_label = [[1, 2], [3, 4], [5, 6]]
        >>> len(data_examples)
        3
        >>> print(data_examples)
        <ListData(
            META INFORMATION
            DATA FIELDS
            Y: [1, 2, 3]
            gt_pseudo_label: [[1, 2], [3, 4], [5, 6]]
            X: [[tensor(1.1949), tensor(-0.9378)], [tensor(0.7414), tensor(0.7603)], [tensor(1.0587), tensor(1.9697)]]
        ) at 0x7f3bbf1991c0>
        >>> print(data_examples[:1])
        <ListData(
            META INFORMATION
            DATA FIELDS
            Y: [1]
            gt_pseudo_label: [[1, 2]]
            X: [[tensor(1.1949), tensor(-0.9378)]]
        ) at 0x7f3bbf1a3580>
        >>> print(data_examples.elements_num("X"))
        6
        >>> print(data_examples.flatten("gt_pseudo_label"))
        [1, 2, 3, 4, 5, 6]
        >>> print(data_examples.to_tuple("Y"))
        (1, 2, 3)
    """

    def __setattr__(self, name: str, value: list):
        """setattr is only used to set data.

        The value must have the attribute of `__len__` and have the same length
        of `ListData`.
        """
        if name in ("_metainfo_fields", "_data_fields"):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(
                    f"{name} has been used as a " "private attribute, which is immutable."
                )

        else:
            # assert isinstance(value, list), "value must be of type `list`"

            # if len(self) > 0:
            #     assert len(value) == len(self), (
            #         "The length of "
            #         f"values {len(value)} is "
            #         "not consistent with "
            #         "the length of this "
            #         ":obj:`ListData` "
            #         f"{len(self)}"
            #     )
            super().__setattr__(name, value)

    __setitem__ = __setattr__

    def __getitem__(self, item: IndexType) -> "ListData":
        """
        Args:
            item (str, int, list, :obj:`slice`, :obj:`numpy.ndarray`,
                :obj:`torch.LongTensor`, :obj:`torch.BoolTensor`):
                Get the corresponding values according to item.

        Returns:
            :obj:`ListData`: Corresponding values.
        """
        assert isinstance(item, IndexType.__args__)
        if isinstance(item, list):
            item = np.array(item)
        if isinstance(item, np.ndarray):
            # The default int type of numpy is platform dependent, int32 for
            # windows and int64 for linux. `torch.Tensor` requires the index
            # should be int64, therefore we simply convert it to int64 here.
            # More details in https://github.com/numpy/numpy/issues/9464
            item = item.astype(np.int64) if item.dtype == np.int32 else item
            item = torch.from_numpy(item)

        if isinstance(item, str):
            return getattr(self, item)

        new_data = self.__class__(metainfo=self.metainfo)

        if isinstance(item, torch.Tensor):
            assert item.dim() == 1, "Only support to get the" " values along the first dimension."

            for k, v in self.items():
                if v is None:
                    new_data[k] = None
                elif isinstance(v, torch.Tensor):
                    new_data[k] = v[item]
                elif isinstance(v, np.ndarray):
                    new_data[k] = v[item.cpu().numpy()]
                elif isinstance(v, (str, list, tuple)) or (
                    hasattr(v, "__getitem__") and hasattr(v, "cat")
                ):
                    # convert to indexes from BoolTensor
                    if isinstance(item, BoolTypeTensor.__args__):
                        indexes = torch.nonzero(item).view(-1).cpu().numpy().tolist()
                    else:
                        indexes = item.cpu().numpy().tolist()
                    slice_list = []
                    if indexes:
                        for index in indexes:
                            slice_list.append(slice(index, None, len(v)))
                    else:
                        slice_list.append(slice(None, 0, None))
                    r_list = [v[s] for s in slice_list]
                    if isinstance(v, (str, list, tuple)):
                        new_value = r_list[0]
                        for r in r_list[1:]:
                            new_value = new_value + r
                    else:
                        new_value = v.cat(r_list)
                    new_data[k] = new_value
                else:
                    raise ValueError(
                        f"The type of `{k}` is `{type(v)}`, which has no "
                        "attribute of `cat`, so it does not "
                        "support slice with `bool`"
                    )

        else:
            # item is a slice or int
            for k, v in self.items():
                if v is None:
                    new_data[k] = None
                else:
                    new_data[k] = v[item]
        return new_data  # type:ignore

    def flatten(self, item: str) -> List:
        """
        Flatten the list of the attribute specified by ``item``.

        Parameters
        ----------
        item
            Name of the attribute to be flattened.

        Returns
        -------
        list
            The flattened list of the attribute specified by ``item``.
        """
        return flatten_list(self[item])

    def elements_num(self, item: str) -> int:
        """
        Return the number of elements in the attribute specified by ``item``.

        Parameters
        ----------
        item : str
            Name of the attribute for which the number of elements is to be determined.

        Returns
        -------
        int
            The number of elements in the attribute specified by ``item``.
        """
        return len(self.flatten(item))

    def to_tuple(self, item: str) -> tuple:
        """
        Convert the attribute specified by ``item`` to a tuple.

        Parameters
        ----------
        item : str
            Name of the attribute to be converted.

        Returns
        -------
        tuple
            The attribute after conversion to a tuple.
        """
        return to_hashable(self[item])

    def __len__(self) -> int:
        """int: The length of ListData."""
        iterator = iter(self._data_fields)
        data = next(iterator)

        while getattr(self, data) is None:
            try:
                data = next(iterator)
            except StopIteration:
                break
        if getattr(self, data) is None:
            raise ValueError("All data fields are None.")
        else:
            return len(getattr(self, data))
