from rolexboost import RolexBoostClassifier, FlexBoostClassifier, RotationForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier


def flexboost_adaboost_test():
    """When K=1, flexboost should converge to adaboost"""

    N_ROW = 1000
    N_COL = 10
    N_REDUNDANT = 1
    N_INFORMATIVE = N_COL - N_REDUNDANT

    TRAIN_RATIO = 0.2
    TRAIN_ROW = int(N_ROW * TRAIN_RATIO)

    TEST_ROUND = 20

    accs = []
    b_accs = []
    for _ in range(TEST_ROUND):
        X, y = make_classification(N_ROW, N_COL, n_informative=N_INFORMATIVE, n_redundant=N_REDUNDANT, n_repeated=0)

        train_X, test_X = X[:TRAIN_ROW], X[TRAIN_ROW:]
        train_y, test_y = y[:TRAIN_ROW], y[TRAIN_ROW:]

        clf = FlexBoostClassifier(n_estimators=100, K=1)
        clf.fit(train_X, train_y)
        pred_y = clf.predict(test_X)
        acc = accuracy_score(test_y, pred_y)
        accs.append(acc)

        # b refers to benchmark
        b_clf = AdaBoostClassifier(n_estimators=100)
        b_clf.fit(train_X, train_y)
        b_pred_y = b_clf.predict(test_X)
        b_acc = accuracy_score(test_y, b_pred_y)
        b_accs.append(b_acc)

    assert np.mean(acc) >= np.mean(b_acc) - 0.02
