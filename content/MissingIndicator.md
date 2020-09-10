---
title: Missing Indicator
tags: machine learning
layout: post
date: 2019-09-30 00:33:00 -0400
comments: true
category: blog
description:
lang: en

---

# Missing Indicator

---
## Overview

This post will demonstrate the use of the `MissingIndicator`, showcasing the behaviour of the algorithm by altering parameters such as "features" and "missing_values" to make sense of the corresponding output.

The docstring reads:

> Binary indicators for missing values.

> Note that this component typically should not be used in a vanilla Pipeline consisting of transformers and a classifier, but rather could be added using a FeatureUnion or ColumnTransformer

The MissingIndicator class is typically used to transform a dataset into its corresponding binary matrix to help indicate the presence of missing values in the dataset.

This transformation is useful in conjunction with (before) imputation. When using imputation, preserving prior information about which values had been missing can be informative.

A `NaN` is the de facto placeholder for missing values and enforces the data type for that feature to be a float/object. However the parameter missing_values can allow the specification of other missing value placeholders such as integers i.e. $-1$ to be identified by the algorithm and transformed.

----
## The Important Parameters

> missing_values : number, string, np.nan (default) or None

> The placeholder for the missing values. All occurrences of `missing_values` will be indicated (True in the output array), the other values will be marked as False.

Specify the indicator for missing values in the dataset here if different from conventional `NaN`.

> features : str, optional

> Whether the imputer mask should represent all or a subset of features.
- If "missing-only" (default), the imputer mask will only represent features containing missing values during fit time.
- If "all", the imputer mask will represent all features.

Change to "all" if the desired output has to retain all features of the dataset. Otherwise algorithm will only keep features that contain missing values.

> error_on_new : boolean, optional

> If True (default), transform will raise an error when there are features with missing values in transform that have no missing values in fit. This is applicable only when ``features="missing-only"``.

Make sure the features with missing values are consistent at the fit and transform stages, otherwise this will output an error message.

---
## Parameters: `missing_values = -1` & `features = "all"`

With these parameter settings, we can expect an array of the same shape as the input to be output. The "$-1$" in the array will be set to `True`.


```python
#  the necessary imports for this exercise
from sklearn.impute import MissingIndicator
import numpy as np
```


```python
x1 = np.array([[9, -1, 5],
               [-1, 5, 9],
               [7, -1, -1]])
x1
```




    array([[ 9, -1,  5],
           [-1,  5,  9],
           [ 7, -1, -1]])




```python
# instantiate an instance of MissingIndicator with missing_values set to "-1"
mi_x1 = MissingIndicator(missing_values=-1, features="all")
x1_tr = mi_x1.fit_transform(x1)
x1_tr
```




    array([[False,  True, False],
           [ True, False, False],
           [False,  True,  True]])



---
## Parameters: `missing_values = np.nan` & `features = "all"`

With these parameter settings, we can expect an array of the same shape as the input to be output. The "np.nan" in the array will be set to `True`.


```python
x2 = np.array([[9, np.nan, 5],
               [np.nan, 5, np.nan],
               [7, -1, -1]])
x2
```




    array([[ 9., nan,  5.],
           [nan,  5., nan],
           [ 7., -1., -1.]])




```python
# instantiate an instance of MissingIndicator with missing_values set to "-1"
mi_x2 = MissingIndicator(missing_values=np.nan, features="all")
x2_tr = mi_x2.fit_transform(x2)
x2_tr
```




    array([[False,  True, False],
           [ True, False,  True],
           [False, False, False]])



---
## Parameters: `missing_values = np.nan` & `features = "missing-only"`

With these parameter settings, we expect an array of a different shape to the input to be output if there are no missing values on some features. The "np.nan" in the array will be set to `True`.

In this case there will be no missing values in the second feature and only missing values on the second row of first and third columns.

We should see an array with the remaining 2 columns transformed as a result.


```python
x3 = np.array([[9, 20, 5],
               [np.nan, 5, np.nan],
               [7, -1, -1]])
x3
```




    array([[ 9., 20.,  5.],
           [nan,  5., nan],
           [ 7., -1., -1.]])




```python
x3_tr = indicator.transform(x3)
x3_tr
```




    array([[False, False],
           [ True,  True],
           [False, False]])



---
## Conclusion

The MissingIndicator while easy to implement, typically results in biased estimates as the transformed data significantly loses interpretability due to the binary nature of the outcome.

While it is decent for a classifier algorithm to perform baseline estimates on, it is generally recommended that other advanced methods or imputers be used to account for missing values to ensure better results.
