from rolexboost import RolexBoostClassifier, FlexBoostClassifier, RotationForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification


def example_test():
    """Test the classifier with randomly generated datasets,
    the accuracy should be better than 60%
    """

    N_ROW = 500
    N_COL = 4
    N_INFORMATIVE = N_COL

    TRAIN_RATIO = 0.2
    TRAIN_ROW = int(N_ROW * TRAIN_RATIO)

    TEST_ROUND = 5

    for _ in range(TEST_ROUND):
        X, y = make_classification(N_ROW, N_COL, n_informative=N_INFORMATIVE, n_redundant=0, n_repeated=0)

        train_X, test_X = X[:TRAIN_ROW], X[TRAIN_ROW:]
        train_y, test_y = y[:TRAIN_ROW], y[TRAIN_ROW:]

        for clf_class in [RolexBoostClassifier, FlexBoostClassifier, RotationForestClassifier]:
            clf = clf_class()
            clf.fit(train_X, train_y)
            pred_y = clf.predict(test_X)

            acc = accuracy_score(test_y, pred_y)
            print(acc)
            assert acc >= 0.6
