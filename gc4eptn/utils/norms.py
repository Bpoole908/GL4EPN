import pandas as pd
import numpy as np

from gc4eptn.utils.utils import filter_df


def feature_norm(X, norm_fn, features):
    X = X.copy()
    assert isinstance(X, pd.DataFrame)
    for feat in features:
        feat_cols = filter_df(X, feat)
        if len(feat_cols) == 0:
            continue
        feat_X = X[feat_cols].values
        X[feat_cols] = norm_fn(feat_X, axis=None)
    return X


def center(X, axis=0):
    if isinstance(X, pd.DataFrame):
        X = X.values
    # print(X.mean(axis=axis, keepdims=True))
    return X - X.mean(axis=axis, keepdims=True)


def standardize(X, axis=0):
    if isinstance(X, pd.DataFrame):
        X = X.values
    # print(X.std(axis=axis, keepdims=True))
    std =  X.std(axis=axis, keepdims=True) + np.finfo(float).eps
    return center(X, axis=axis) / std


def minmax(X, axis=0):
    if isinstance(X, pd.DataFrame):
        X = X.values
    X_min = X.min(axis=axis)
    X_max = X.max(axis=axis)
    return (X - X_min) / (X_max - X_min)
