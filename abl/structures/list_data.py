# Copyright (c) OpenMMLab. All rights reserved.
import itertools
from typing import List, Union

import numpy as np
import torch

from ..utils import flatten as flatten_list
from ..utils import to_hashable
from .base_data_element import BaseDataElement

BoolTypeTensor = Union[torch.BoolTensor, torch.cuda.BoolTensor]
LongTypeTensor = Union[torch.LongTensor, torch.cuda.LongTensor]

IndexType = Union[str, slice, int, list, LongTypeTensor, BoolTypeTensor, np.ndarray]


# Modified from
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/data_structures/instance_data.py # noqa
class ListData(BaseDataElement):
    """Data structure for instance-level annotations or predictions.

    Subclass of :class:`BaseDataElement`. All value in `data_fields`
    should have the same length. This design refer to
    https://github.com/facebookresearch/detectron2/blob/master/detectron2/structures/instances.py # noqa E501
    ListData also support extra functions: ``index``, ``slice`` and ``cat`` for data field. The type of value
    in data field can be base data structure such as `torch.Tensor`, `numpy.ndarray`, `list`, `str`, `tuple`,
    and can be customized data structure that has ``__len__``, ``__getitem__`` and ``cat`` attributes.

    Examples:
        >>> # custom data structure
        >>> class TmpObject:
        ...     def __init__(self, tmp) -> None:
        ...         assert isinstance(tmp, list)
        ...         self.tmp = tmp
        ...     def __len__(self):
        ...         return len(self.tmp)
        ...     def __getitem__(self, item):
        ...         if isinstance(item, int):
        ...             if item >= len(self) or item < -len(self):  # type:ignore
        ...                 raise IndexError(f'Index {item} out of range!')
        ...             else:
        ...                 # keep the dimension
        ...                 item = slice(item, None, len(self))
        ...         return TmpObject(self.tmp[item])
        ...     @staticmethod
        ...     def cat(tmp_objs):
        ...         assert all(isinstance(results, TmpObject) for results in tmp_objs)
        ...         if len(tmp_objs) == 1:
        ...             return tmp_objs[0]
        ...         tmp_list = [tmp_obj.tmp for tmp_obj in tmp_objs]
        ...         tmp_list = list(itertools.chain(*tmp_list))
        ...         new_data = TmpObject(tmp_list)
        ...         return new_data
        ...     def __repr__(self):
        ...         return str(self.tmp)
        >>> from mmengine.structures import ListData
        >>> import numpy as np
        >>> import torch
        >>> img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
        >>> instance_data = ListData(metainfo=img_meta)
        >>> 'img_shape' in instance_data
        True
        >>> instance_data.det_labels = torch.LongTensor([2, 3])
        >>> instance_data["det_scores"] = torch.Tensor([0.8, 0.7])
        >>> instance_data.bboxes = torch.rand((2, 4))
        >>> instance_data.polygons = TmpObject([[1, 2, 3, 4], [5, 6, 7, 8]])
        >>> len(instance_data)
        2
        >>> print(instance_data)
        <ListData(
            META INFORMATION
            img_shape: (800, 1196, 3)
            pad_shape: (800, 1216, 3)
            DATA FIELDS
            det_labels: tensor([2, 3])
            det_scores: tensor([0.8000, 0.7000])
            bboxes: tensor([[0.4997, 0.7707, 0.0595, 0.4188],
                        [0.8101, 0.3105, 0.5123, 0.6263]])
            polygons: [[1, 2, 3, 4], [5, 6, 7, 8]]
        ) at 0x7fb492de6280>
        >>> sorted_results = instance_data[instance_data.det_scores.sort().indices]
        >>> sorted_results.det_scores
        tensor([0.7000, 0.8000])
        >>> print(instance_data[instance_data.det_scores > 0.75])
        <ListData(
            META INFORMATION
            img_shape: (800, 1196, 3)
            pad_shape: (800, 1216, 3)
            DATA FIELDS
            det_labels: tensor([2])
            det_scores: tensor([0.8000])
            bboxes: tensor([[0.4997, 0.7707, 0.0595, 0.4188]])
            polygons: [[1, 2, 3, 4]]
        ) at 0x7f64ecf0ec40>
        >>> print(instance_data[instance_data.det_scores > 1])
        <ListData(
            META INFORMATION
            img_shape: (800, 1196, 3)
            pad_shape: (800, 1216, 3)
            DATA FIELDS
            det_labels: tensor([], dtype=torch.int64)
            det_scores: tensor([])
            bboxes: tensor([], size=(0, 4))
            polygons: []
        ) at 0x7f660a6a7f70>
        >>> print(instance_data.cat([instance_data, instance_data]))
        <ListData(
            META INFORMATION
            img_shape: (800, 1196, 3)
            pad_shape: (800, 1216, 3)
            DATA FIELDS
            det_labels: tensor([2, 3, 2, 3])
            det_scores: tensor([0.8000, 0.7000, 0.8000, 0.7000])
            bboxes: tensor([[0.4997, 0.7707, 0.0595, 0.4188],
                        [0.8101, 0.3105, 0.5123, 0.6263],
                        [0.4997, 0.7707, 0.0595, 0.4188],
                        [0.8101, 0.3105, 0.5123, 0.6263]])
            polygons: [[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4], [5, 6, 7, 8]]
        ) at 0x7f203542feb0>
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

    @staticmethod
    def cat(instances_list: List["ListData"]) -> "ListData":
        """Concat the instances of all :obj:`ListData` in the list.

        Note: To ensure that cat returns as expected, make sure that
        all elements in the list must have exactly the same keys.

        Args:
            instances_list (list[:obj:`ListData`]): A list
                of :obj:`ListData`.

        Returns:
            :obj:`ListData`
        """
        assert all(isinstance(results, ListData) for results in instances_list)
        assert len(instances_list) > 0
        if len(instances_list) == 1:
            return instances_list[0]

        # metainfo and data_fields must be exactly the
        # same for each element to avoid exceptions.
        field_keys_list = [instances.all_keys() for instances in instances_list]
        assert len({len(field_keys) for field_keys in field_keys_list}) == 1 and len(
            set(itertools.chain(*field_keys_list))
        ) == len(field_keys_list[0]), (
            "There are different keys in "
            "`instances_list`, which may "
            "cause the cat operation "
            "to fail. Please make sure all "
            "elements in `instances_list` "
            "have the exact same key."
        )

        new_data = instances_list[0].__class__(metainfo=instances_list[0].metainfo)
        for k in instances_list[0].keys():
            values = [results[k] for results in instances_list]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                new_values = torch.cat(values, dim=0)
            elif isinstance(v0, np.ndarray):
                new_values = np.concatenate(values, axis=0)
            elif isinstance(v0, (str, list, tuple)):
                new_values = v0[:]
                for v in values[1:]:
                    new_values += v
            elif hasattr(v0, "cat"):
                new_values = v0.cat(values)
            else:
                raise ValueError(
                    f"The type of `{k}` is `{type(v0)}` which has no " "attribute of `cat`"
                )
            new_data[k] = new_values
        return new_data  # type:ignore

    def flatten(self, item: IndexType) -> List:
        """Flatten self[item].

        Returns:
            list: Flattened data fields.
        """
        return flatten_list(self[item])

    def elements_num(self, item: IndexType) -> int:
        """int: The number of elements in self[item]."""
        return len(self.flatten(item))

    def to_tuple(self, item: IndexType) -> tuple:
        """tuple: The data fields in self[item] converted to tuple."""
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
