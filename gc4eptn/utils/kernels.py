import numpy as np
from sklearn.gaussian_process.kernels import RationalQuadratic

from gc4eptn.utils.utils import get_df_values

def corr(X, **kwargs):
    X = get_df_values(X)
    return np.corrcoef(X, **kwargs)


def cov(X, **kwargs):
    X = get_df_values(X)
    return np.cov(X, **kwargs)


def gram(X, transpose=False): 
    X = get_df_values(X)
    return X @ X.T if transpose else X.T @ X


def rational_quadratic(X, **kwargs):
    X = get_df_values(X)
    rq = RationalQuadratic(**kwargs)
    return rq(X)


def weighted_kernel(df, kernel, columns=['mag', 'ang'], weights=[0.5, 0.5], return_all=False):
    assert np.sum(weights) == 1
    assert len(columns) == len(weights)
    
    wmat = None
    k_dfs = {}
    for col, w in zip(columns, weights):
        col_df = df[filter_df(df, [col])]
        k = kernel(col_df.values)
        if wmat is None:
            wmat = w * k
        else:
            wmat += w * k
        k_dfs[col] = k
        
    if return_all:
        return wmat, k_dfs
    else:
        return wmat

