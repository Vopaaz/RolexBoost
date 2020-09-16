from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.tree import DecisionTreeClassifier

__all__ = ["RotationForestClassifier", "FlexBoostClassifier", "RolexBoostClassifier"]


class RotationForestClassifier(BaseEstimator, ClassifierMixin):
    """Temperoral implementation that use a DecisionTreeClassifier to mock the classifier behavior"""

    def __init__(self):
        super()
        self._inner_estimator = DecisionTreeClassifier()

    def fit(self, X, y):
        self._inner_estimator.fit(X, y)
        return self

    def predict(self, X):
        return self._inner_estimator.predict(X)


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
