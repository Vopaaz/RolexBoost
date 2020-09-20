from rolexboost import RotationForestClassifier, RolexBoostClassifier, FlexBoostClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
import numpy as np
import pandas as pd
from tests.data import load_data, data_attributes
import time
from nose.plugins.attrib import attr

APPROX_THRESH = 0.9
N_JOBS = -1


def get_best_params(estimator, X, y):
    grid = {
        "max_depth": [1, 3, None],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf": [1, 2, 3],
    }
    if isinstance(estimator, (RolexBoostClassifier, FlexBoostClassifier)):
        grid = {**{"K": np.arange(0.3, 1.0, 5e-2)}, **grid}
    cv = GridSearchCV(estimator, grid, n_jobs=N_JOBS, refit=False).fit(X, y)
    return cv.best_params_


def five_fold_CV(estimator, X, y):
    CV = 5
    res = cross_validate(estimator, X, y, scoring="accuracy", cv=CV, n_jobs=N_JOBS)["test_score"].sum() / CV
    return res


def twenty_times_CV(estimator, X, y):
    ROUNDS = 20
    res = sum(five_fold_CV(estimator, X, y) for _ in range(ROUNDS)) / ROUNDS
    return res

@attr("slow")
def performance_tests():
    results = []
    for dataset, benchmark in data_attributes.items():
        X, y = load_data(dataset)

        def test_one_algorithm(algorithm_name, clf_class):
            # Grid Search
            clf = clf_class()
            best_params = get_best_params(clf, X, y)

            # Cross Validation
            clf = clf_class(**best_params)
            acc = twenty_times_CV(clf, X, y)

            # Report
            return acc

        for algorithm, clf_class in zip(["rotation", "flex", "rolex"], [RotationForestClassifier, FlexBoostClassifier, RolexBoostClassifier]):
            acc = test_one_algorithm(algorithm, clf_class)
            algorithm_benchmark = benchmark[algorithm]
            results.append((dataset, algorithm, acc, algorithm_benchmark))

    df = pd.DataFrame(results, columns=["dataset", "algorithm", "accuracy", "benchmark"])
    df["ratio"] = df["accuracy"] / df["benchmark"]
    df.to_csv("tests/latest-performance-report-detail.csv", index=False)

    grouped_df = df.groupby("algorithm").agg("mean")
    grouped_df["ratio"] = grouped_df["accuracy"] / grouped_df["benchmark"]
    grouped_df.to_csv("tests/latest-performance-report-aggregated.csv", index=True)
    assert (grouped_df["ratio"] >= APPROX_THRESH).all()
