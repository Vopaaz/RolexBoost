from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from rolexboost.util import split_subsets, bootstrap, rearrange_matrix_row, ensemble_predictions
from rolexboost.exceptions import NotFittedException, InsufficientDataException
from rolexboost.lib import PCA
import numpy as np
import scipy

__all__ = ["RotationForestClassifier", "FlexBoostClassifier", "RolexBoostClassifier"]


class RotationForestClassifier(BaseEstimator, ClassifierMixin):
    """Temperoral implementation that use a DecisionTreeClassifier to mock the classifier behavior"""

    def __init__(
        self,
        n_estimators=100,
        n_features_per_subset=3,  # In the algorithm description, the parameter is the number of subspaces.
        # However, in the validation part, "the number of features in each subset was set to three".
        # The parameter is thus formulated as number of features per subset, to make the future reproduction of evaluation easier
        bootstrap_rate=0.75,
        **decision_tree_kwargs
    ):
        super()
        self.n_estimators = n_estimators
        self.n_features_per_subset = n_features_per_subset
        self.bootstrap_rate = bootstrap_rate
        self.decision_tree_kwargs = decision_tree_kwargs

    def fit(self, X, y):
        if X.shape[0] < self.n_features_per_subset:
            raise InsufficientDataException(self.n_features_per_subset.X.shape[0])

        self.estimators = [self._fit_one_estimator(X, y) for _ in range(self.n_estimators)]
        return self

    def _fit_one_estimator(self, X, y):
        idx, X_subsets = split_subsets(X, self.n_features_per_subset)
        X_bootstrapped = [bootstrap(x, self.bootstrap_rate) for x in X_subsets]
        pca_coefficients = [PCA().fit(x).components_ for x in X_bootstrapped]
        raw_diag_matrix = scipy.linalg.block_diag(*pca_coefficients)
        rotation_matrix = rearrange_matrix_row(raw_diag_matrix, np.concatenate(idx))
        rotated_X = X.dot(rotation_matrix)

        clf = DecisionTreeClassifier(**self.decision_tree_kwargs)
        clf.fit(rotated_X, y)
        clf._rotation_matrix = rotation_matrix
        return clf

    def predict(self, X):
        if not hasattr(self, "estimators"):
            raise NotFittedException(self)

        predictions = [clf.predict(X.dot(clf._rotation_matrix)) for clf in self.estimators]
        return ensemble_predictions(predictions)


class FlexBoostClassifier(BaseEstimator, ClassifierMixin):
    """Temperoral implementation that use a DecisionTreeClassifier to mock the classifier behavior"""

    def __init__(self):
        super()
        self._inner_estimator = DecisionTreeClassifier()

    def fit(self, X, y):
        self._inner_estimator.fit(X, y)
        return self

    def predict(self, X):
        return self._inner_estimator.predict(X)


class RolexBoostClassifier(BaseEstimator, ClassifierMixin):
    """Temperoral implementation that use a DecisionTreeClassifier to mock the classifier behavior"""

    def __init__(self):
        super()
        self._inner_estimator = DecisionTreeClassifier()

    def fit(self, X, y):
        self._inner_estimator.fit(X, y)
        return self

    def predict(self, X):
        return self._inner_estimator.predict(X)
