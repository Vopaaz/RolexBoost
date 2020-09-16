from rolexboost.util import split_subsets
import numpy as np


def split_subsets_test():
    X = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
    idx, val = split_subsets(X, 2)
    assert len(idx) == 3
    assert len(val) == 3

    for index, value in zip(idx, val):
        assert (X[:, index] == value).all()
