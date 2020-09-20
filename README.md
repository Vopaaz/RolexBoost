# RolexBoost

Unofficial implementation of D. Yang, H. Lee and D. Lim, "RolexBoost: A Rotation-Based Boosting Algorithm With Adaptive Loss Functions," in IEEE Access, vol. 8, pp. 41037-41044, 2020, doi: 10.1109/ACCESS.2020.2976822.

This is the course project of Fundamentals of Machine Learning, Tsinghua University, 2020 Autumn.

## Installation

```bash
pip install rolexboost
```

## API reference

We provided scikit-learn-like API for the RolexBoost algorithm proposed in the paper,
together with the RotationForest and FlexBoost, which are the source of RolexBoost's idea.

Note that
1. Only classifiers are provided. We did not implemented the regressors because they are not mentioned in the paper, while this project is intended as a reproduction.
2. We only ensures that the `fit` and `predict` API works well. Some others, such as `score`, may be functional thanks to the scikit-learn `BaseEstimator` and `ClassifierMixin` base classes, but still others, such as `fit_predict` or `predict_proba`, are currently unavailable.

We might implement those two in the future if someone is interested in this project.

### Basic Example

```python
>>> import pandas as pd
>>> import numpy as np
>>> from rolexboost import RolexBoostClassifier, FlexBoostClassifier, RotationForestClassifier

>>> clf = RolexBoostClassifier() # Or the other two classifiers

>>> df = pd.DataFrame({"A": [2,6,5,7,1,8], "B":[8,5,2,3,4,6], "C": [3,9,5,4,6,1], "Label": [0,1,1,0,0,1]})
>>> df
   A  B  C  Label
0  2  8  3      0
1  6  5  9      1
2  5  2  5      1
3  7  3  4      0
4  1  4  6      0
5  8  6  1      1

>>> X = df[["A", "B", "C"]]
>>> y = df["Label"]

>>> clf.fit(X, y)
RolexBoostClassifier()
>>> clf.predict(X)
array([0, 1, 1, 0, 0, 1], dtype=int64)

>>> test_X = np.array([
...     [3,1,2],
...     [2,5,1],
...     [5,1,7]
... ])

>>> clf.predict(test_X)
array([1, 0, 1], dtype=int64)
```

### API References

#### Rotation Forest

```python
RotationForestClassifier(
    n_estimators=100,
    n_features_per_subset=3,
    bootstrap_rate=0.75,
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
)
```

- `n_estimators`: number of base estimators
- `n_features_per_subset`: number of features in each subset
- `bootstrap_rate`: ratio of samples bootstrapped in the original dataset

All other parameters are passed to the `DecisionTreeClassifier` of scikit-learn. Please refer to [their documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier) for details.


Note:

In the algorithm description, a parameter controls the number of subsets, and the number of features is derived from it.
However, the validation part of the paper says that "the number of features in each subset was set to three".
In our framework, the parameter `n_samples_per_subset` is thus formulated in this way to make the benchmark evaluation easier.


#### FlexBoost

```python
FlexBoostClassifier(
    n_estimators=100,
    K=0.5,
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
)
```

- `n_estimators`: number of base estimators
- `K`: the parameter to control the "aggressiveness" and "conservativeness" in the adaptive loss function choosing process. It should be a number between 0 and 1.

All other parameters are passed to the `DecisionTreeClassifier` of scikit-learn. Please refer to [their documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier) for details.

The default parameter for `max_depth` is 1, because FlexBoost is a modification of AdaBoost, and they should converge to the same result when `K=1`.
In [scikit-learn implementation of AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier), the default `max_depth` for the DecisionTreeClassifier is 1.



#### RolexBoost

```python
RolexBoostClassifier(
    n_estimators=100,
    n_features_per_subset=3,
    bootstrap_rate=0.75,
    K=0.5,
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
    ccp_alpha=0.0
)
```

- `n_estimators`: number of base estimators
- `n_features_per_subset`: number of features in each subset
- `bootstrap_rate`: ratio of samples bootstrapped in the original dataset
- `K`: the parameter to control the "aggressiveness" and "conservativeness" in the adaptive loss function choosing process. It should be a number between 0 and 1.


All other parameters are passed to the `DecisionTreeClassifier` of scikit-learn. Please refer to [their documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier) for details.


Note:

In the algorithm description, a parameter controls the number of subsets, and the number of features is derived from it.
However, the validation part of the paper says that "the number of features in each subset was set to three".
In our framework, the parameter `n_samples_per_subset` is thus formulated in this way to make the benchmark evaluation easier.

The default parameter for `max_depth` is 1, because RolexBoost integrates the idea of FlexBoost. Please refer to the last section about why FlexBoost has a default `max_depth` of 1.


## Performance Benchmarks





