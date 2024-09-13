import os
import functools
import inspect
from pathlib import Path
from pdb import set_trace

import pandas as pd
import numpy as np

import gc4eptn 


def get_module_root():
    return Path(gc4eptn.__file__).parent


def extract_dir_path(path, dir):
    if not isinstance(path, Path):
        path = Path(path)
    parts = path.parts
    dir_loc = np.where( np.array(parts) == dir)[0]
    if len(dir_loc) == 0:
        raise ValueError(f"No directory {dir} could be found within the path {path}")
    else:
        dir_loc = dir_loc[0]
    return Path(os.path.join(*parts[:dir_loc+1]))


def build_experiment_path(subdirs, path=None):
    if path is None:
        path = get_module_root().parent / 'exps' 
    for d in subdirs:
        path = path / str(d)
    path.mkdir(parents=True, exist_ok=True)
    return path


def filter_df(df, regex):
        if not isinstance(regex, (list, tuple)):
            regex = [regex]
        found = [list(df.filter(regex=r).columns) for r in regex]
        return np.hstack(found) if len(found) != 0 else np.array([], dtype=object)


def recursive_getattr(obj, path: str, delim='.', default=None):
    """
    :param obj: Object
    :param path: 'attr1.attr2.etc'
    :param default: Optional default value, at any point in the path
    :return: obj.attr1.attr2.etc
    """
    attrs = path.split('.')
    try:
        return functools.reduce(getattr, attrs, obj)
    except AttributeError:
        if default:
            return default
        raise AttributeError


def get_dataloader(dataset_type, dataset_version, difficulty):
    class_name = f'{difficulty}PMUData{dataset_type}{dataset_version}'
    for key  in gc4eptn.dataloaders.__dict__.keys():
        if key.startswith('__'):
            continue
        if class_name.lower() == key.lower():
            return getattr(gc4eptn.dataloaders, key)
       
    raise ValueError(f"Class {class_name.lower()} was not found.")


def get_df_values(x):
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.values
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise TypeError(f"Passed data `x` is neither a pandas object or ndarray")
    

def float_exponent_notation(float_number, precision_digits=6, format_type="g"):
    """
        Returns a string representation of the scientific notation of the given number 
        formatted for use with LaTeX or Mathtext, with `precision_digits` digits of
        mantissa precision, printing a normal decimal if an exponent isn't necessary.
    """
    e_float = "{0:.{1:d}{2}}".format(float_number, precision_digits, format_type)
    if "e" not in e_float:
        return "${}$".format(e_float)
    mantissa, exponent = e_float.split("e")
    cleaned_exponent = exponent.strip("+")
    return "${0} \\times 10^{{{1}}}$".format(mantissa, cleaned_exponent)


def to_adjacency(x, zero_diag=True):
    x = x.copy()
    x[x != 0] = 1
    if zero_diag:
        np.fill_diagonal(x, 0)
    return x.astype(int)


def thresholding(A, tau):
    """Thresholding array A by tau"""
    mask = (np.abs(A) > tau)
    np.fill_diagonal(mask,1.) # dont threshold on diagonal
    return A*mask


def filter_function_kwargs(fn, **kwargs):
    sig = inspect.signature(fn)
    filter_keys = [param.name for param in sig.parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD]
    filtered_dict = {key:kwargs[key] for key in filter_keys if key in kwargs}
    return filtered_dict


def lambda_grid_R_max_min(R, n_samples, n_lambdas=50, round_dec=None):
    I_p = np.identity(len(R)) 
    max = np.max(R - I_p)
    min = np.min(R - I_p)
    
    max_R = np.max([max, -min])
    min_R =  0.0001 * max_R
    print(min_R, max_R)
    return exp_log_grid(min_R, max_R, n_lambdas, round_dec)*n_samples


def lambda_grid_R_max_square(R, n_samples, n_lambdas=100, min_R=0.01, round_dec=None):
    I_p = np.identity(len(R)) * np.diagonal(R)
    max = np.max(R - I_p)
    min = np.min(R - I_p)
    
    max_R = np.max([max, -min])*n_samples**2
    return exp_log_grid(min_R, max_R, n_lambdas, round_dec)


def exp_log_grid(start, end, n=100, round_dec=None):
    grid = np.exp(np.linspace(np.log(start), np.log(end), n))
    if round_dec is not None:
        grid = np.round(grid, round_dec)
    return grid
