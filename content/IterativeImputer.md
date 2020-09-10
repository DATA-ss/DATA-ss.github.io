---
title: Iterative Imputer
tags: machine learning
layout: post
date: 2019-09-30 00:35:00 -0400
comments: true
category: blog
description:
lang: en

---

# Iterative Imputer

The `IterativeImputer` is an estimator that's still experimental as of the time of writing (27 September 2019).

The docstring reads:

> Multivariate imputer that estimates each feature from all the others.

> A strategy for imputing missing values by modeling each feature with missing values as a function of other features in a round-robin fashion.

This means that the algorithm will take any feature with missing values and treat it like a pseudo-target, in turn using the other columns as its' features. The order of selected columns can be altered via its' parameters.

Therefore its important to note that this process would introduce some measure of correlation to the data.

Because the algorithm is using a regressor as an estimator to fill the columns with missing values, we will also want to ensure that each feature is as normally distributed as possible. This means that transformations may need to be performed to ensure optimal results.

----
## The Important Parameters

> estimator: default=BayesianRidge()

> `The estimator to use at each step of the round-robin imputation. If sample_posterior is True, the estimator must support return_std in its predict method.`

Uses BayesianRidge by default. Can be swapped out for any other regressor such as KNeighborsRegressor or DecisionTreesRegressor among others.
Make sure to define the parameters of the regressor when calling it as an estimator for the algorithm.

> missing_values : int, np.nan, optional (default=np.nan)

> `The placeholder for the missing values. All occurrences of missing_values will be imputed.`

Use this to inform the algorithm as to how the missing values are defined in the dataset.

> sample_posterior : boolean, default=False

> `Whether to sample from the (Gaussian) predictive posterior of the fitted estimator for each imputation. Estimator must support return_std in its predict method if set to True. Set to True if using IterativeImputer for multiple imputations.`

This parameter is usually usable only when a Bayesian regressor is used as an estimator.

> imputation_order : str, optional (default="ascending")
> The order in which the features will be imputed. Possible values:
- "ascending" - From features with fewest missing values to most.
- "descending" - From features with most missing values to fewest.
- "roman" - Left to right.
- "arabic" - Right to left.
- "random" - A random order for each round.

Use this option to alter the order for which features get imputed first, as the round-robin nature of the algorithm means different results depending on the setting of this parameter.

---
## Overview

This blog shall serve as a proof of concept to demonstrate the usage of the `IterativeImputer` as a viable strategy for filling in missing values within any feature columns.

We will follow this procedure to test out the `IterativeImputer`:

1) Create a regression set.

2) Insert NaNs into regression set's features.

3) Impute NaNs with `IterativeImputer` first, followed by the `SimpleImputer`. `IterativeImputer` will go through the `KNeighborsRegressor`, `LinearRegression` and `BayesianRidge` estimators.

4) Fit the regression set to an `ExtraTreesRegressor`.

5) Predict the target using the `IterativeImputer` and `SimpleImputer` sets and compare MSE scores.

---
## 1) Make a Regression Set


```python
# the import packages we'll need
from sklearn.experimental import enable_iterative_imputer # this step is necessary because the IterativeImputer is still experimental
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.metrics import mean_squared_error
```


```python
X, y, coef = make_regression(n_samples=100, n_features=5, noise=0.1, coef=True, random_state=42)
```


```python
X = pd.DataFrame(X, columns=["feature_1", "feature_2","feature_3","feature_4","feature_5"])
```

## 2) Insert NaNs


```python
np.random.seed(42)
XX = X.copy()

# fill a copy of the dataset with NaNs at random
for i in np.random.choice((range(len(XX.index))),size=20, replace=False):
    for col in np.random.choice(XX.columns,size=1):
        XX.loc[i+np.random.choice(range(10)), col] = np.NaN
        XX.loc[i+np.random.choice(range(10)), col] = np.NaN
```

## 3) Impute NaNs with `IterativeImputer` and `SimpleImputer`


```python
# instantiate an instance of the IterativeImputer with the KNeighborsRegressor as an estimator
ii_kn = IterativeImputer(estimator=KNeighborsRegressor(n_neighbors=15), verbose=2, max_iter=1000)
# Set verbose=2 to see the workflow of the IterativeImputer
XX_ii_kn= ii_kn.fit_transform(XX)
```

    [IterativeImputer] Completing matrix with shape (100, 5)
    [IterativeImputer] Ending imputation round 1/1000, elapsed time 0.01
    [IterativeImputer] Ending imputation round 2/1000, elapsed time 0.02
    [IterativeImputer] Ending imputation round 3/1000, elapsed time 0.03
    [IterativeImputer] Ending imputation round 4/1000, elapsed time 0.03
    [IterativeImputer] Early stopping criterion reached.



```python
# instantiate an instance of the IterativeImputer with the LinearRegression as an estimator
ii_lin = IterativeImputer(estimator=LinearRegression(), verbose=2)
# Set verbose=2 to see the workflow of the IterativeImputer
XX_ii_lin= ii_lin.fit_transform(XX)
```

    [IterativeImputer] Completing matrix with shape (100, 5)
    [IterativeImputer] Ending imputation round 1/10, elapsed time 0.04
    [IterativeImputer] Ending imputation round 2/10, elapsed time 0.04
    [IterativeImputer] Ending imputation round 3/10, elapsed time 0.05
    [IterativeImputer] Early stopping criterion reached.



```python
# instantiate an instance of the IterativeImputer with the BayesianRidge as an estimator
ii_br = IterativeImputer(estimator=BayesianRidge(), verbose=2)
# Set verbose=2 to see the workflow of the IterativeImputer
XX_ii_br= ii_br.fit_transform(XX)
```

    [IterativeImputer] Completing matrix with shape (100, 5)
    [IterativeImputer] Ending imputation round 1/10, elapsed time 0.01
    [IterativeImputer] Ending imputation round 2/10, elapsed time 0.03
    [IterativeImputer] Early stopping criterion reached.



```python
si = SimpleImputer()
XX_si = si.fit_transform(XX)
```

## 4) Fit to a `ExtraTreesRegressor`


```python
et = ExtraTreesRegressor(n_estimators=100, random_state=42)
et.fit(X,y)
```




    ExtraTreesRegressor(bootstrap=False, criterion='mse', max_depth=None,
                        max_features='auto', max_leaf_nodes=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=1, min_samples_split=2,
                        min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
                        oob_score=False, random_state=42, verbose=0,
                        warm_start=False)



## 5) Predicting the Targets and Scoring


```python
yhat = et.predict(X)
print(f"MSE for base generated model: {mean_squared_error(y, yhat)}")
```

    MSE for base generated model: 4.78746936230924e-26



```python
yhat_ii_lin = et.predict(XX_ii_lin)
mean_squared_error(y, yhat_ii_lin)
print(f"MSE for IterativeImputer with LinearRegression estimator: {mean_squared_error(y, yhat_ii_lin)}")
```

    MSE for IterativeImputer with LinearRegression estimator: 1060.1077381402583



```python
yhat_ii_kn = et.predict(XX_ii_kn)
mean_squared_error(y, yhat_ii_kn)
print(f"MSE for IterativeImputer with KNeighborsRegressor estimator: {mean_squared_error(y, yhat_ii_kn)}")
```

    MSE for IterativeImputer with KNeighborsRegressor estimator: 855.3049715708974



```python
yhat_ii_br = et.predict(XX_ii_br)
mean_squared_error(y, yhat_ii_br)
print(f"MSE for IterativeImputer with BayesianRidge estimator: {mean_squared_error(y, yhat_ii_br)}")
```

    MSE for IterativeImputer with BayesianRidge estimator: 1039.619040088303



```python
yhat_si = et.predict(XX_si)
mean_squared_error(y, yhat_si)
print(f"MSE for SimpleImputer: {mean_squared_error(y, yhat_si)}")
```

    MSE for SimpleImputer: 1040.8413193768447


# Conclusion

As we can see from above, the `Mean_Squared_Errors` scores demonstate that the IterativeImputer yields superior results to the SimpleImputer. The single exception being when the simple LinearRegression was used as an estimator (let's just not use the basic LinearRegression at this point!).

We should take note that the `IterativeImputer` is still in its' developmental stage and may still be subject to bugs and errors. Nevertheless the early indications are that it'll become another useful tool in any data scientist's arsenal to handle missing values.
