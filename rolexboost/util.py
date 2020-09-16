from sklearn.tree import DecisionTreeClassifier
import numpy as np
import random


def get_CART_tree():
    return DecisionTreeClassifier()


def split_subsets(X, n_features_per_subset):
    """
    Returns: (idx, value)
    - idx: a list of 1-d array, whose element is the index of the subset of columns
    - value: a list of 2-d array, the actual splitted values

    Example:
    ```python
    >>> X = np.array([
    ...    [1,2,3,4,5],
    ...    [6,7,8,9,10],
    ...    [11,12,13,14,15]
    ... ])
    >>> idx, val = split_subsets(X, 2)
    >>> idx
    [
        [0,3],
        [1,2],
        [4]
    ]
    >>> val
    [
        [[1,4], [6,9], [11,14]],
        [[2,3], [7,8], [12,13]],
        [[5], [10], [15]]
    ]
    ```

    Note that
    1. the function is random, so the provided output is just a possible outcome.
    2. a list of list is provided as the return value here only to improve readability.
        The type of the output is a list of 1-d numpy array.
    """

    n_features = X.shape[1]
    all_index = list(range(n_features))
    np.random.shuffle(all_index)

    n_subsets = int(np.ceil(n_features / n_features_per_subset))
    idx, val = [], []
    for i in range(n_subsets):
        this_index = all_index[i * n_features_per_subset : (i + 1) * n_features_per_subset]
        idx.append(this_index)
        val.append(X[:, this_index])

    return idx, val


def bootstrap(X, ratio):
    return np.random.choice(X, size=int(X.shape[0] * ratio))
