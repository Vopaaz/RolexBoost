from rolexboost.util import split_subsets, rearrange_matrix_row, ensemble_predictions_unweighted, ensemble_predictions_weighted
import numpy as np


def split_subsets_test():
    X = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
    idx, val = split_subsets(X, 2)
    assert len(idx) == 3
    assert len(val) == 3

    for index, value in zip(idx, val):
        assert (X[:, index] == value).all()


def rearrange_matrix_row_test():
    mat = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    idx = [2, 1, 3, 0]
    res = rearrange_matrix_row(mat, idx)
    expected = np.array([[9, 10, 11], [3, 4, 5], [0, 1, 2], [6, 7, 8]])
    assert (res == expected).all()


def ensemble_predictions_unweighted_test():
    predictions = [np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), np.array([1, 0, 0, 1])]
    res = ensemble_predictions_unweighted(predictions)
    expected = np.array([0, 0, 1, 1])
    assert (res == expected).all()


def ensemble_predictions_weighted_test():
    predictions = [np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0]), np.array([1, 1, 0, 0]), np.array([0, 0, 1, 0])]
    weights = [10, 0.1, 0.1, 9.9]
    res = ensemble_predictions_weighted(predictions, weights)
    expected = np.array([0, 0, 1, 0])
    assert (res == expected).all()
