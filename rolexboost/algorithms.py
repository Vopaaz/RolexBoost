from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from rolexboost.util import split_subsets, bootstrap, rearrange_matrix_row, ensemble_predictions_unweighted, ensemble_predictions_weighted
from rolexboost.exceptions import NotFittedException, InsufficientDataException
from rolexboost.lib import PCA
import numpy as np
import scipy

__all__ = ["RotationForestClassifier", "FlexBoostClassifier", "RolexBoostClassifier"]


class RotationForestClassifier(BaseEstimator, ClassifierMixin):
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
        return ensemble_predictions_unweighted(predictions)


class FlexBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, K=0.5, **decision_tree_kwargs):
        super()
        self.n_estimators = n_estimators
        self.K = K
        self.decision_tree_kwargs = {**{"max_depth": 1}, **decision_tree_kwargs}

        self.ERROR_THRESHOLD = 1e-100

    def calc_error(self, y_true, y_pred, weight):
        return np.average(y_true != y_pred, weights=weight)

    def calc_alpha(self, k, error):
        EPSILON = 1e-100
        return 1 / (2 * k) * np.log((1 - error + EPSILON) / (error + EPSILON))

    def _fit_one_estimator(self, X, y, previous_weight=None, previous_error=None, previous_alpha=None, previous_prediction=None):
        """
        Returns: (DecisionTreeClassifier, weight, error, alpha, prediction)
        """
        if previous_weight is None and previous_error is None and previous_prediction is None:
            length = X.shape[0]
            weight = np.full((length,), 1 / length)

            clf = DecisionTreeClassifier(**self.decision_tree_kwargs)
            clf.fit(X, y, sample_weight=weight)

            prediction = clf.predict(X)
            error = self.calc_error(y, prediction, weight)
            alpha = self.calc_alpha(1, error)

            return clf, weight, error, alpha, prediction

        best_clf, best_weight, best_alpha, best_prediction = None, None, None, None
        best_error = np.inf

        print()
        for k in [self.K, 1, 1 / self.K]:
            clf = DecisionTreeClassifier(**self.decision_tree_kwargs)
            weight = previous_weight * np.exp(
                k * previous_alpha * np.vectorize(lambda y_true, y_pred: -1 if y_true == y_pred else 1)(y, previous_prediction)
            )
            weight /= weight.sum()
            clf.fit(X, y, sample_weight=weight)
            prediction = clf.predict(X)
            error = self.calc_error(y, prediction, weight)
            print(k, error)
            if error < best_error:
                best_error = error
                best_clf, best_weight, best_alpha, best_prediction = clf, weight, self.calc_alpha(k, error), prediction
        print(best_error, best_alpha)
        print()

        return best_clf, best_weight, best_error, best_alpha, best_prediction

    def fit(self, X, y):
        weight, error, alpha, prediction = None, None, None, None
        self.estimators = []
        self.alphas = []
        for i in range(self.n_estimators):
            clf, weight, error, alpha, prediction = self._fit_one_estimator(X, y, weight, error, alpha, prediction)
            self.estimators.append(clf)
            self.alphas.append(alpha)
            if error <= self.ERROR_THRESHOLD:
                break
        return self

    def predict(self, X):
        predictions = [clf.predict(X) for clf in self.estimators]
        return ensemble_predictions_weighted(predictions, self.alphas)


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
