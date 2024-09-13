import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List, Union
from pdb import set_trace

import pandas as pd
import numpy as np
import networkx as nx
from networkx.drawing import layout
from gglasso.helper.data_generation import generate_precision_matrix, sample_covariance_matrix
from gglasso.helper.basic_linalg import adjacency_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class SyntheticGraphData():
    """
    Generates a sparse precision matrix with associated covariance matrix from a random network.
    
    
    Parameters
    ----------
    p : int, optional
        size of the matrix. The default is 100.
    N : int, optional
        number of samples to samples true covariance in order to generate the
        sample covariance.
    M : int, optional
        number of subblocks. p/M must result in an integer. The default is 10.
    style : str, optional
        Type of the random network. Available network types:
            * 'powerlaw': a powerlaw network.
            * 'erdos': a Erdos-Renyi network.
        The default is 'powerlaw'.
    gamma : float, optional
        parameter for powerlaw network. The default is 2.8.
    prob : float, optional
        probability of edge creation for Erdos-Renyi network. The default is 0.1.
    scale : boolean, optional
        whether Sigma (cov. matrix) is scaled by diagonal entries (as described by Danaher et al.). If set to True, then the generated precision matrix is not
        the inverse of Sigma anymore.
    seed : int, optional
        Seed for network creation and matrix entries. The default is None.
    """
            
    def __init__(
        self,  
        p,
        M=1, 
        style = 'powerlaw', 
        gamma = 2.8, 
        prob = 0.1, 
        scale = False, 
        seed = None,
        layout_fn=layout.circular_layout,
    ):
        self.p = p
        self.seed = seed
        
        self.Sigma, self.Theta = generate_precision_matrix(
            p=p, 
            M=M, 
            style=style, 
            gamma=gamma, 
            prob=prob, 
            scale=scale, 
            seed=seed
        )
        self.A = adjacency_matrix(self.Theta)
        self.G = nx.from_numpy_array(self.A)
        self.labels = {n:str(n) for n in np.arange(len(self.A))}
        self.set_positions(layout_fn)

    def load_graph_data(self, N, return_df: bool = False, return_S: bool = False):
        S, X = sample_covariance_matrix(self.Sigma, N)
        if return_df:
            df = pd.DataFrame(X, index=self.labels.values())
            if return_S:
                return df, S
            return df
        if return_S:
            return X, S
        return X
    
    def set_positions(self, layout_fn, **kwargs):
        self.graph_positions = layout_fn(self.G, **kwargs)

    def get_edge_locations(self):
        return np.where(self.A != 0)
    
    def plot_ground_truth(self):
        nx.draw_networkx(
            self.G, 
            pos=self.graph_positions, 
            node_color="darkblue", 
            edge_color="darkblue", 
            font_color='white', 
            with_labels=True
        )