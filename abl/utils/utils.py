import numpy as np
from itertools import chain


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
    if not isinstance(nested_list, list):
        raise TypeError("Input must be of type list.")

    if not nested_list or not isinstance(nested_list[0], (list, tuple)):
        return nested_list

    return list(chain.from_iterable(nested_list))


def reform_idx(flattened_list, structured_list):
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


def block_sample(X, Z, Y, sample_num, seg_idx):
    """
    Extract a block of samples from lists X, Z, and Y.

    Parameters
    ----------
    X, Z, Y : list
        Input lists from which to extract the samples.
    sample_num : int
        The number of samples per block.
    seg_idx : int
        The block index to extract.

    Returns
    -------
    tuple of lists
        The extracted block samples from X, Z, and Y.

    Example
    -------
    >>> X = [1, 2, 3, 4, 5, 6]
    >>> Z = ['a', 'b', 'c', 'd', 'e', 'f']
    >>> Y = [10, 11, 12, 13, 14, 15]
    >>> block_sample(X, Z, Y, 2, 1)
    ([3, 4], ['c', 'd'], [12, 13])
    """
    start_idx = sample_num * seg_idx
    end_idx = sample_num * (seg_idx + 1)

    return (data[start_idx:end_idx] for data in (X, Z, Y))


def check_equal(a, b, max_err=0):
    """
    Check whether two numbers a and b are equal within a maximum allowable error.

    Parameters
    ----------
    a, b : int or float
        The numbers to compare.
    max_err : int or float, optional
        The maximum allowable absolute difference between a and b for them to be considered equal.
        Default is 0, meaning the numbers must be exactly equal.

    Returns
    -------
    bool
        True if a and b are equal within the allowable error, False otherwise.

    Raises
    ------
    TypeError
        If a or b are not of type int or float.
    """
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        raise TypeError("Input values must be int or float.")

    return abs(a - b) <= max_err


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


def hashable_to_list(x):
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
        return [hashable_to_list(item) for item in x]
    return x


def calculate_revision_num(parameter, total_length):
    """
    Convert a float parameter to an integer, based on a total length.

    Parameters
    ----------
    parameter : int or float
        The parameter to convert. If float, it should be between 0 and 1.
        If int, it should be non-negative. If -1, it will be replaced with total_length.
    total_length : int
        The total length to calculate the parameter from if it's a fraction.

    Returns
    -------
    int
        The calculated parameter.

    Raises
    ------
    TypeError
        If parameter is not an int or a float.
    ValueError
        If parameter is a float not in [0, 1] or an int below 0.
    """
    if not isinstance(parameter, (int, float)):
        raise TypeError("Parameter must be of type int or float.")

    if parameter == -1:
        return total_length
    elif isinstance(parameter, float):
        if not (0 <= parameter <= 1):
            raise ValueError("If parameter is a float, it must be between 0 and 1.")
        return round(total_length * parameter)
    else:
        if parameter < 0:
            raise ValueError("If parameter is an int, it must be non-negative.")
        return parameter


if __name__ == "__main__":
    A = np.array(
        [
            [
                0.18401675,
                0.06797526,
                0.06797541,
                0.06801736,
                0.06797528,
                0.06797526,
                0.06818808,
                0.06797527,
                0.06800033,
                0.06797526,
                0.06797526,
                0.06797526,
                0.06797526,
            ],
            [
                0.07223161,
                0.0685229,
                0.06852708,
                0.17227574,
                0.06852163,
                0.07018146,
                0.06860291,
                0.06852849,
                0.06852163,
                0.0685216,
                0.0685216,
                0.06852174,
                0.0685216,
            ],
            [
                0.06794382,
                0.0679436,
                0.06794395,
                0.06794346,
                0.06794346,
                0.18467231,
                0.06794345,
                0.06794871,
                0.06794345,
                0.06794345,
                0.06794345,
                0.06794345,
                0.06794345,
            ],
        ],
        dtype=np.float32,
    )
    B = [[0, 9, 3], [0, 11, 4]]

    print(ori_confidence_dist(A, B))
    print(confidence_dist(A, B))
