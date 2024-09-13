from typing import (
    Dict,
    Callable
)   
from pathlib import Path
from os.path import join
from functools import partial
from pdb import set_trace

import numpy as np
import matplotlib.pyplot as plt
from gglasso.helper.model_selection import thresholding
from gc4eptn.utils.metrics import (
    graph_fscore,
    ebic,
    hamming_distance
)

import rpy2.robjects as ro
from rpy2.robjects import r, pandas2ri,numpy2ri
from rpy2.robjects.packages import importr
from rpy2.rinterface_lib.embedded import RRuntimeError

from gc4eptn.utils.plotting import plot_recovered
from gc4eptn.utils.utils import to_adjacency, filter_function_kwargs


class GGMncv:
    """ Wraps R's GGMncv library  for running non-convex GGM """
    
    def __init__(self):
        self.ggmncv = importr('GGMncv')
        self.base = importr('base')
        self.as_dataframe = r('as.data.frame')

    def __call__(self, R, n, penalty, *, verbose=False, **kwargs):
        """ Runs the ggmncv R function
        
            Note: 
                See function arguments https://www.rdocumentation.org/packages/GGMncv/versions/2.1.1/topics/ggmncv
        """
        exception = None
        try:
            with (ro.default_converter + numpy2ri.converter).context():
                results = self.ggmncv.ggmncv(R, n, penalty, **kwargs)
        except RRuntimeError as e:
            exception = e
        finally:   
            if verbose:
                print(self.base.warnings())
            if exception is not None:
                raise RRuntimeError(exception)
        return results

    def _parse_results(self, results):
        res_dict = {}
        names = results.names
        if not isinstance(names, np.ndarray) and names == ro.rinterface.NULL:
            return results
        
        for n in names:
            value = results.rx2[str(n)]
            if hasattr(value, 'rx2'):
                res_dict[n] = self._parse_results(value)
            else:
                res_dict[n] = value
        return res_dict
    
    def df_to_r(self, df):
        """ Converts a Python DataFrame to a R DataFrame"""
        with (ro.default_converter + pandas2ri.converter).context():
            rdf = ro.conversion.py2rpy(df)
        return rdf
    
    def r_to_df(self, rdf):
        """ Converts a R DataFrame to a Python DataFrame"""
        with (ro.default_converter + pandas2ri.converter).context():
            df = ro.conversion.rpy2py(rdf)
        return df

    def array2d_to_r(self, array):
        assert len(array.shape) == 2
        nrow, ncol = array.shape
        with (ro.default_converter + numpy2ri.converter).context():
            matrix = ro.r.matrix(array, nrow=nrow, ncol=ncol)
        return matrix


def find_best_model(metric_fn, A, R, n_samples, results, select_metric='lowest', ):
    best = {}
    for name, model in results['fitted_models'].items():
        kwargs = filter_function_kwargs(
            metric_fn,
            A=A,
            A_hat=to_adjacency(model['wi']),
            theta=model['wi'],
            R=R,
            N=n_samples
        )
        score = metric_fn(**kwargs) 
        if select_metric == 'lowest':
            score_measure = score <= best.get('score', np.inf)
        elif select_metric == 'highest':
            score_measure = score >=  best.get('score', -np.inf)
        else:
            raise ValueError("`select_metric` {select_metric} must be either 'lowest' or 'highest'")
        
        if score_measure:
            # print(score, score_measure, best.get('score'))s
            best['score'] = score 
            best['model'] = model
            best['name'] = name
    
    return best


def single_ggm_exp(
    R,
    A,
    penalty,
    n_samples,
    positions,
    metric_fn=None,
    select_metric='lowest',
    *,
    tau=None,
    labels=None,
    annot=True,
    save_dir_path=None,
    file_prefix='',
    save_ext='pdf',
    ggmncv_kwargs: Dict = None,
    verbose: bool = False,
    disable_plotting: bool = False
):
    save_path = None
    ggmncv_kwargs = ggmncv_kwargs or {}
    
    ggmncv = GGMncv()
    results = ggmncv(R=R, n=n_samples, penalty=penalty, verbose=verbose, **ggmncv_kwargs)
    if metric_fn is not None:
        best = find_best_model(
            R=R, 
            A=A,
            n_samples=n_samples,
            results=results, 
            metric_fn=metric_fn, 
            select_metric=select_metric
        )
        model, score = best['model'], best['score']
    else:
        model = results['fit']
        score = model['ic'][0]

    theta_hat = model['wi']
    if tau is not None: 
        theta_hat = thresholding(theta_hat, tau=tau)
    A_hat = to_adjacency(theta_hat)
    ic =  model['ic'][0]
    hd = hamming_distance(A, A_hat)
    lambda_ = model['lambda'][0]
    gamma = model['gamma'][0]
    
    info = {
        'metrics': {
            'ic': ic,
            'score': score,
            'hd': hd
        },
        'params': {
            'lambda': lambda_, 
            'gamma': gamma,
        }
    }
    
    title = f"{penalty} - Score: {np.round(score, 3)} HD: {hd} IC: {np.round(ic, 3)} Lambda: {np.round(lambda_, 3)} Gamma: {np.round(gamma, 3):.3f}"
    title += f"Tau: {np.round(tau,5)}" if tau is not None else ''
    file_name = f'{file_prefix}-' if len(file_prefix) != 0  else ''
    file_name += f"{penalty}_l={np.round(lambda_, 3)}_gamma={gamma}"
    file_name += f"_tau={np.round(tau,2)}" if tau is not None else ''
    if save_dir_path is not None:
        save_dir_path = save_dir_path/penalty
        save_dir_path.mkdir(parents=True, exist_ok=True)
        save_path = save_dir_path/f'{file_name}.{save_ext}'

    plot = plot_recovered(
        A=A_hat, 
        theta=theta_hat,
        positions=positions, 
        labels=labels,
        title=title, 
        sub_titles=['Recovered Graph', 'Recovered Precision Matrix'],
        save_path=save_path,
        number_fmt='.2g',
        annot_kws = {"fontsize":9} ,
        vmin=None,
        vmax=None,
        annot=annot,
    )
    
    if not disable_plotting: 
        plt.show()
    
    return A_hat, theta_hat, info, plot, results