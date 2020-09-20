from sklearn.tree import DecisionTreeClassifier
import numpy as np
import random
from rolexboost.exceptions import DimNotMatchException
import scipy
from collections import defaultdict


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
    idx = np.random.randint(X.shape[0], size=int(X.shape[0] * ratio))
    return X[idx]


def rearrange_matrix_row(mat, idx):
    """Rearrange the row of the matrix according to the given index

    Parameters:
    - mat: the matrix to convert
    - idx: an iterable indicating the current arrangement of the rows of the matrix.
            The rows are expected to be swapped so that the corresponding idx will be in increasing order

    Example:
    ```python
    >>> mat = np.array([
        [0,1,2],
        [3,4,5],
        [6,7,8],
        [9,10,11]
    ])
    >>> idx = [2,1,3,0]
    >>> rearrange_matrix_row(mat, idx)
    np.array([
        [9,10,11],
        [3,4,5],
        [0,1,2],
        [6,7,8]
    ])
    ```
    """
    n_rows = mat.shape[0]
    if n_rows != len(idx):
        raise DimNotMatchException(n_rows, len(idx))

    # The first element is the row's current (physical) index, the second element is its ideal index
    row_indexes = [(i, index) for i, index in enumerate(idx)]
    # Sort by the ideal index, so that by slicing with the rearranged physical index, we get the rearranged matrix
    sorted_row_indexes = sorted(row_indexes, key=lambda x: x[1])
    # Do the slicing
    indexes = [i[0] for i in sorted_row_indexes]
    return mat[indexes]


def ensemble_predictions_unweighted(predictions):
    """Ensemble the predictions using mode.
    Parameter:
    - predictions: list of 1-d array, each representing the prediction of a base estimator

    Example:
    ```python
    >>> predictions = [
    ...     np.array([0,0,1,1]),
    ...     np.array([0,1,1,1]),
    ...     np.array([1,0,0,1])
    ... ]
    >>> ensemble_predictions_unweighted(predictions)
    np.array([0,0,1,1])
    ```
    """

    pred_arr = np.stack(predictions, axis=-1)
    return scipy.stats.mode(pred_arr, axis=1)[0].reshape(-1)


def ensemble_predictions_weighted(predictions, weights):
    """Ensemble the predictions according to the weights.
    Parameter:
    - predictions: list of 1-d array, each representing the prediction of a base estimator
    - weights: list of numbers, each representing the weight of the corresponding prediction

    Example:
    ```python
    >>> predictions = [
    ...     np.array([0,0,1,1]),
    ...     np.array([1,1,0,0]),
    ...     np.array([1,1,0,0]),
    ...     np.array([0,0,1,0])
    ... ]
    >>> weights = [10, 0.1, 0.1, 9.9]
    >>> ensemble_predictions_weighted(predictions, weights)
    np.array([0,0,1,0])
    ```
    """

    length = predictions[0].shape[0]
    acc = [{} for _ in range(length)]
    for prediction, weight in zip(predictions, weights):
        for ix in range(length):
            this_ix_pred = prediction[ix]
            acc[ix][this_ix_pred] = acc[ix].get(this_ix_pred, 0) + weight
    result = [max(d.items(), key=lambda t: t[1])[0] for d in acc]
    return np.array(result)


EPSILON = 1e-100

K_ALPHA_THRESHOLD = 500


def calc_error(y_true, y_pred, weight):
    return np.average(y_true != y_pred, weights=weight)


def calc_alpha(k, error):
    return 1 / (2 * k) * np.log((1 - error + EPSILON) / (error + EPSILON))


def calc_updated_weight(weight, k, alpha, y_true, y_pred):
    """The returned weight is normalized"""
    new_weights = weight * np.exp(k * alpha * np.vectorize(lambda y_t, y_p: 0 if y_t == y_p else 1)(y_true, y_pred))
    return new_weights / new_weights.sum()


def as_numpy_array(*objs):
    if len(objs) == 1:
        return np.asarray(objs[0])
    else:
        return (np.asarray(o) for o in objs)
