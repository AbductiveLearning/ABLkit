from itertools import chain

import numpy as np


def flatten(nested_list):
    """
    Flattens a nested list.

    Parameters
    ----------
    nested_list : list
        A list which might contain sublists or tuples.

    Returns
    -------
    list
        A flattened version of the input list.

    Raises
    ------
    TypeError
        If the input object is not a list.
    """
    # if not isinstance(nested_list, list):
    #     raise TypeError("Input must be of type list.")

    if isinstance(nested_list, list) and len(nested_list) == 0:
        return nested_list

    if not isinstance(nested_list, list) or not isinstance(nested_list[0], (list, tuple)):
        return nested_list

    return list(chain.from_iterable(nested_list))


def reform_list(flattened_list, structured_list):
    """
    Reform the index based on structured_list structure.

    Parameters
    ----------
    flattened_list : list
        A flattened list of predictions.
    structured_list : list
        A list containing saved predictions, which could be nested lists or tuples.

    Returns
    -------
    list
        A reformed list that mimics the structure of structured_list.
    """
    # if not isinstance(flattened_list, list):
    #     raise TypeError("Input must be of type list.")

    if not isinstance(structured_list[0], (list, tuple)):
        return flattened_list

    reformed_list = []
    idx_start = 0
    for elem in structured_list:
        idx_end = idx_start + len(elem)
        reformed_list.append(flattened_list[idx_start:idx_end])
        idx_start = idx_end

    return reformed_list


def hamming_dist(pred_pseudo_label, candidates):
    """
    Compute the Hamming distance between two arrays.

    Parameters
    ----------
    pred_pseudo_label : list
        First array to compare.
    candidates : list
        Second array to compare, expected to have shape (n, m)
        where n is the number of rows, m is the length of pred_pseudo_label.

    Returns
    -------
    numpy.ndarray
        Hamming distances.
    """
    pred_pseudo_label = np.array(pred_pseudo_label)
    candidates = np.array(candidates)

    # Ensuring that pred_pseudo_label is broadcastable to the shape of candidates
    pred_pseudo_label = np.expand_dims(pred_pseudo_label, 0)

    return np.sum(pred_pseudo_label != candidates, axis=1)


def confidence_dist(pred_prob, candidates):
    """
    Compute the confidence distance between prediction probabilities and candidates.

    Parameters
    ----------
    pred_prob : list of numpy.ndarray
        Prediction probability distributions, each element is an ndarray
        representing the probability distribution of a particular prediction.
    candidates : list of list of int
        Index of candidate labels, each element is a list of indexes being considered
        as a candidate correction.

    Returns
    -------
    numpy.ndarray
        Confidence distances computed for each candidate.
    """
    pred_prob = np.clip(pred_prob, 1e-9, 1)
    _, cols = np.indices((len(candidates), len(candidates[0])))
    return 1 - np.prod(pred_prob[cols, candidates], axis=1)


def to_hashable(x):
    """
    Convert a nested list to a nested tuple so it is hashable.

    Parameters
    ----------
    x : list or other type
        A potentially nested list to convert to a tuple.

    Returns
    -------
    tuple or other type
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
    x : tuple or other type
        A potentially nested tuple to convert to a list.

    Returns
    -------
    list or other type
        The input converted to a list if it was a tuple,
        otherwise the original input.
    """
    if isinstance(x, tuple):
        return [restore_from_hashable(item) for item in x]
    return x


