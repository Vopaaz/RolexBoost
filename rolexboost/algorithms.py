from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from rolexboost.util import (
    split_subsets,
    bootstrap,
    rearrange_matrix_row,
    ensemble_predictions_unweighted,
    ensemble_predictions_weighted,
    calc_alpha,
    calc_error,
    calc_updated_weight,
    K_ALPHA_THRESHOLD,
    as_numpy_array,
)
from rolexboost.exceptions import NotFittedException, InsufficientDataException
from sklearn.decomposition import PCA
import numpy as np
import scipy
from abc import ABC

__all__ = ["RotationForestClassifier", "FlexBoostClassifier", "RolexBoostClassifier"]


class RolexAlgorithmMixin(BaseEstimator, ClassifierMixin):
    def _check_fitted(self):
        if not hasattr(self, "estimators_"):
            raise NotFittedException(self)

    def _get_decision_tree_classifier(self):
        return DecisionTreeClassifier(
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            class_weight=self.class_weight,
            random_state=self.random_state,
            min_impurity_decrease=self.min_impurity_decrease,
            min_impurity_split=self.min_impurity_split,
            presort=self.presort,
            ccp_alpha=self.ccp_alpha,
        )


class RotationForestClassifier(RolexAlgorithmMixin):
    def __init__(
        self,
        n_estimators=100,
        n_features_per_subset=3,  # In the algorithm description, the parameter is the number of subspaces.
        # However, in the validation part, "the number of features in each subset was set to three".
        # The parameter is thus formulated as number of features per subset, to make the future reproduction of evaluation easier
        bootstrap_rate=0.75,
        # DecisionTreeClassifier parameters
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        class_weight=None,
        presort="deprecated",
        ccp_alpha=0.0,
    ):
        self.n_estimators = n_estimators
        self.n_features_per_subset = n_features_per_subset
        self.bootstrap_rate = bootstrap_rate
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.class_weight = class_weight
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.presort = presort
        self.ccp_alpha = ccp_alpha

    def _rotation_precheck(self, X):
        if X.shape[0] < self.n_features_per_subset:
            raise InsufficientDataException(self.n_features_per_subset.X.shape[0])

    def fit(self, X, y):
        X, y = as_numpy_array(X, y)
        self._rotation_precheck(X)
        self.estimators_ = [self._fit_one_estimator(X, y) for _ in range(self.n_estimators)]
        return self

    def _construct_rotation_matrix(self, X):
        idx, X_subsets = split_subsets(X, self.n_features_per_subset)
        X_bootstrapped = [bootstrap(x, self.bootstrap_rate) for x in X_subsets]
        pca_coefficients = [PCA().fit(x).components_ for x in X_bootstrapped]
        raw_diag_matrix = scipy.linalg.block_diag(*pca_coefficients)
        rotation_matrix = rearrange_matrix_row(raw_diag_matrix, np.concatenate(idx))
        return rotation_matrix

    def _fit_one_estimator(self, X, y):
        rotation_matrix = self._construct_rotation_matrix(X)
        rotated_X = X.dot(rotation_matrix)

        clf = self._get_decision_tree_classifier()
        clf.fit(rotated_X, y)
        clf._rotation_matrix = rotation_matrix

        return clf

    def predict(self, X):
        self._check_fitted()
        X = as_numpy_array(X)
        predictions = [clf.predict(X.dot(clf._rotation_matrix)) for clf in self.estimators_]
        return ensemble_predictions_unweighted(predictions)


class FlexBoostClassifier(RolexAlgorithmMixin):
    def __init__(
        self,
        n_estimators=100,
        K=0.5,
        # DecisionTreeClassifier parameters
        criterion="gini",
        splitter="best",
        max_depth=1,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        class_weight=None,
        presort="deprecated",
        ccp_alpha=0.0,
    ):
        self.n_estimators = n_estimators
        self.K = K
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.class_weight = class_weight
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.presort = presort
        self.ccp_alpha = ccp_alpha

    def _fit_first_estimator(self, X, y):
        length = X.shape[0]
        weight = np.full((length,), 1 / length)

        clf = self._get_decision_tree_classifier()
        clf.fit(X, y, sample_weight=weight)

        prediction = clf.predict(X)
        error = calc_error(y, prediction, weight)
        alpha = calc_alpha(1, error)

        return clf, weight, error, alpha, prediction

    def _fit_subsequent_estimator(self, X, y, previous_weight, previous_error, previous_alpha, previous_prediction):
        best_clf, best_weight, best_alpha, best_prediction = None, None, None, None
        best_eval_error = np.inf
        best_k = None

        for k in [1, self.K, 1 / self.K]:

            weight = calc_updated_weight(previous_weight, k, previous_alpha, y, previous_prediction)
            clf = self._get_decision_tree_classifier()
            clf.fit(X, y, sample_weight=weight)
            prediction = clf.predict(X)

            # For choose from the three k's, use the same previous weight, otherwise they are not comparable
            error = calc_error(y, prediction, previous_weight)
            if error < best_eval_error:
                best_eval_error = error
                best_clf, best_weight, best_prediction = clf, weight, prediction
                best_k = k

        # When one is selected as the best, the error passed to the next round should use its own weight.
        best_error = calc_error(y, best_prediction, best_weight)
        best_alpha = calc_alpha(best_k, best_error)
        return best_clf, best_weight, best_error, best_alpha, best_prediction

    def _fit_one_estimator(self, X, y, previous_weight=None, previous_error=None, previous_alpha=None, previous_prediction=None):
        """
        Returns: (DecisionTreeClassifier, weight, error, alpha, prediction)
        """
        if previous_weight is None and previous_error is None and previous_prediction is None:
            return self._fit_first_estimator(X, y)
        else:
            return self._fit_subsequent_estimator(X, y, previous_weight, previous_error, previous_alpha, previous_prediction)

    def fit(self, X, y):
        X, y = as_numpy_array(X, y)

        weight, error, alpha, prediction = None, None, None, None
        self.estimators_ = []
        self.alphas = []
        for i in range(self.n_estimators):
            clf, weight, error, alpha, prediction = self._fit_one_estimator(X, y, weight, error, alpha, prediction)
            self.estimators_.append(clf)
            self.alphas.append(alpha)
            if 1 / self.K * alpha > K_ALPHA_THRESHOLD:
                break
        return self

    def predict(self, X):
        self._check_fitted()
        X = as_numpy_array(X)

        predictions = [clf.predict(X) for clf in self.estimators_]
        return ensemble_predictions_weighted(predictions, self.alphas)


class RolexBoostClassifier(RotationForestClassifier, FlexBoostClassifier):
    """Temperoral implementation that use a DecisionTreeClassifier to mock the classifier behavior"""

    def __init__(
        self,
        n_estimators=100,
        n_features_per_subset=3,  # See the inline comment for n_features_per_subset of RotationForest constructor
        bootstrap_rate=0.75,
        K=0.5,
        # DecisionTreeClassifier parameters
        criterion="gini",
        splitter="best",
        max_depth=1,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        class_weight=None,
        presort="deprecated",
        ccp_alpha=0.0,
    ):
        self.n_estimators = n_estimators
        self.n_features_per_subset = n_features_per_subset
        self.bootstrap_rate = bootstrap_rate
        self.K = K
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.class_weight = class_weight
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.presort = presort
        self.ccp_alpha = ccp_alpha

    def _fit_one_estimator(self, X, y, previous_weight=None, previous_error=None, previous_alpha=None, previous_prediction=None):
        rotation_matrix = self._construct_rotation_matrix(X)
        rotated_X = X.dot(rotation_matrix)

        if previous_weight is None and previous_error is None and previous_prediction is None:
            clf, weight, error, alpha, prediction = self._fit_first_estimator(rotated_X, y)
        else:
            clf, weight, error, alpha, prediction = self._fit_subsequent_estimator(
                rotated_X, y, previous_weight, previous_error, previous_alpha, previous_prediction
            )
        clf._rotation_matrix = rotation_matrix
        return clf, weight, error, alpha, prediction

    def fit(self, X, y):
        X, y = as_numpy_array(X, y)
        self._rotation_precheck(X)
        FlexBoostClassifier.fit(self, X, y)
        return self

    def predict(self, X):
        self._check_fitted()
        X = as_numpy_array(X)
        predictions = [clf.predict(X.dot(clf._rotation_matrix)) for clf in self.estimators_]
        return ensemble_predictions_weighted(predictions, self.alphas)
