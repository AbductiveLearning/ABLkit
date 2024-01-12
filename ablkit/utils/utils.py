"""
Implementation of utilities used in ablkit.

Copyright (c) 2024 LAMDA.  All rights reserved.
"""

from typing import List, Any, Union, Tuple, Optional

import numpy as np


def flatten(nested_list: List[Union[Any, List[Any], Tuple[Any, ...]]]) -> List[Any]:
    """
    Flattens a nested list at the first level.

    Parameters
    ----------
    nested_list : List[Union[Any, List[Any], Tuple[Any, ...]]]
        A list which might contain sublists or tuples at the first level.

    Returns
    -------
    List[Any]
        A flattened version of the input list, where only the first
        level of sublists and tuples are reduced.
    """
    if not isinstance(nested_list, list):
        return nested_list

    flattened_list = []
    for item in nested_list:
        if isinstance(item, (list, tuple)):
            flattened_list.extend(item)
        else:
            flattened_list.append(item)

    return flattened_list


def reform_list(
    flattened_list: List[Any], structured_list: List[Union[Any, List[Any], Tuple[Any, ...]]]
) -> List[List[Any]]:
    """
    Reform the list based on the structure of ``structured_list``.

    Parameters
    ----------
    flattened_list : List[Any]
        A flattened list of elements.
    structured_list : List[Union[Any, List[Any], Tuple[Any, ...]]]
        A list that reflects the desired structure, which may contain sublists or tuples.

    Returns
    -------
    List[List[Any]]
        A reformed list that mimics the structure of ``structured_list``.
    """
    if not isinstance(structured_list[0], (list, tuple)):
        return flattened_list

    reformed_list = []
    idx_start = 0
    for elem in structured_list:
        idx_end = idx_start + len(elem)
        reformed_list.append(flattened_list[idx_start:idx_end])
        idx_start = idx_end

    return reformed_list


def hamming_dist(pred_pseudo_label: List[Any], candidates: List[List[Any]]) -> np.ndarray:
    """
    Compute the Hamming distance between two arrays.

    Parameters
    ----------
    pred_pseudo_label : List[Any]
        Pseudo-labels of an example.
    candidates : List[List[Any]]
        Multiple possible candidates.

    Returns
    -------
    np.ndarray
        Hamming distances computed for each candidate.
    """
    pred_pseudo_label = np.array(pred_pseudo_label)
    candidates = np.array(candidates)

    # Ensuring that pred_pseudo_label is broadcastable to the shape of candidates
    pred_pseudo_label = np.expand_dims(pred_pseudo_label, 0)

    return np.sum(pred_pseudo_label != candidates, axis=1)


def confidence_dist(pred_prob: np.ndarray, candidates_idxs: List[List[Any]]) -> np.ndarray:
    """
    Compute the confidence distance between prediction probabilities and candidates,
    where the confidence distance is defined as 1 - the product of prediction probabilities.

    Parameters
    ----------
    pred_prob : np.ndarray
        Prediction probability distributions, each element is an array
        representing the probability distribution of a particular prediction.
    candidates_idxs : List[List[Any]]
        Multiple possible candidates' indices.

    Returns
    -------
    np.ndarray
        Confidence distances computed for each candidate.
    """
    pred_prob = np.clip(pred_prob, 1e-9, 1)
    cols = np.arange(len(candidates_idxs[0]))[None, :]
    return 1 - np.prod(pred_prob[cols, candidates_idxs], axis=1)


def avg_confidence_dist(pred_prob: np.ndarray, candidates_idxs: List[List[Any]]) -> np.ndarray:
    """
    Compute the average confidence distance between prediction probabilities and candidates,
    where the confidence distance is defined as 1 - the average of prediction probabilities.

    Parameters
    ----------
    pred_prob : np.ndarray
        Prediction probability distributions, each element is an array
        representing the probability distribution of a particular prediction.
    candidates_idxs : List[List[Any]]
        Multiple possible candidates' indices.

    Returns
    -------
    np.ndarray
        Confidence distances computed for each candidate.
    """
    cols = np.arange(len(candidates_idxs[0]))[None, :]
    return 1 - np.average(pred_prob[cols, candidates_idxs], axis=1)


def to_hashable(x: Union[List[Any], Any]) -> Union[Tuple[Any, ...], Any]:
    """
    Convert a nested list to a nested tuple so it is hashable.

    Parameters
    ----------
    x : Union[List[Any], Any]
        A potentially nested list to convert to a tuple.

    Returns
    -------
    Union[Tuple[Any, ...], Any]
        The input converted to a tuple if it was a list,
        otherwise the original input.
    """
    if isinstance(x, list):
        return tuple(to_hashable(item) for item in x)
    return x


def restore_from_hashable(x):
    """
    Convert a nested tuple back to a nested list.

    Parameters
    ----------
    x : Union[Tuple[Any, ...], Any]
        A potentially nested tuple to convert to a list.

    Returns
    -------
    Union[List[Any], Any]
        The input converted to a list if it was a tuple,
        otherwise the original input.
    """
    if isinstance(x, tuple):
        return [restore_from_hashable(item) for item in x]
    return x


def tab_data_to_tuple(
    X: Union[List[Any], Any], y: Union[List[Any], Any], reasoning_result: Optional[Any] = 0
) -> Tuple[List[List[Any]], List[List[Any]], List[Any]]:
    """
    Convert a tabular data to a tuple by adding a dimension to each element of
    X and y. The tuple contains three elements: data, label, and reasoning result.
    If X is None, return None.

    Parameters
    ----------
    X : Union[List[Any], Any]
        The data.
    y : Union[List[Any], Any]
        The label.
    reasoning_result : Any, optional
        The reasoning result. Defaults to 0.

    Returns
    -------
    Tuple[List[List[Any]], List[List[Any]], List[Any]]
        A tuple of (data, label, reasoning_result).
    """
    if X is None:
        return None
    if len(X) != len(y):
        raise ValueError(
            f"The length of X and y should be the same, but got {len(X)} and {len(y)}."
        )
    return ([[x] for x in X], [[y_item] for y_item in y], [reasoning_result] * len(y))
