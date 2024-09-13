from itertools import product
from pathlib import Path
from typing import (
    Dict,
    List,
    Callable
)
from pdb import set_trace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from gc4eptn.dataloaders import PMUData
from gc4eptn.utils.plotting import plot_heatmap

class KernelAnalysis():
    def __init__(
        self,
        pmuds_class: PMUData, 
        pmuds_kwargs: Dict[str, Dict], 
        build_graph_kwargs: Dict[str, Dict],
        save_dir_path: str = None
    ):
        self.pmuds_class = pmuds_class
        self.pmuds_kwargs = pmuds_kwargs
        self.build_graph_kwargs = build_graph_kwargs
        self.stat_names = ['rank', 'cond']
        self.save_dir_path = save_dir_path if save_dir_path is not None else Path('.')
        
    def __call__(
        self,
        kernel_fn: Callable, 
        title='{0} {1}', 
        verbose=False,
        annot_kws=None,
        fmt='.4f',
        save=False,
        save_filename='{0}-{1}.pdf', 
    ):
        annot_kws = annot_kws or {"fontsize":7}
        R_summary = []
        for pmuds_name, pmuds_kwargs in self.pmuds_kwargs.items():
            R_stats = []
            for build_name, kwargs in self.build_graph_kwargs.items():
                pmuds = self.pmuds_class(**pmuds_kwargs)
                X = pmuds.build_graph_data(**kwargs).graph_df
                R = kernel_fn(X)
                # mask = ~np.eye(R.shape[0],dtype=bool)
                cond_numb = np.linalg.cond(R)
                rank = np.linalg.matrix_rank(R)
                R_stats.append([rank, cond_numb])

                ax = plot_heatmap(
                    R, 
                    xticklabels=pmuds.labels.values(), 
                    yticklabels=pmuds.labels.values(), 
                    fmt=fmt, 
                    annot_kws=annot_kws
                )
                ax.set_title(title.format(pmuds_name, build_name))
                if save:
                    sdp = self.save_dir_path / pmuds_name / build_name 
                    sdp.mkdir(parents=True, exist_ok=True)
                    ax.figure.savefig(sdp / save_filename.format(pmuds_name, build_name))
                plt.show()
                    
                if verbose:
                    print(f"Rank: {rank}")
                    print(f"Condition Number: {cond_numb}")
                    display(
                        pd.DataFrame(
                            X,
                            index=pmuds.graph_df.index, 
                            columns=pmuds.graph_df.columns,
                        )
                    )
                    display(
                        pd.DataFrame(
                            R,
                            index=pmuds.labels.values(), 
                            columns=pmuds.labels.values(),
                        ).style.background_gradient(cmap='Blues')
                    )
            R_summary.append(np.hstack(R_stats))

        multi_index = list(product(self.build_graph_kwargs.keys(), self.stat_names))

        return pd.DataFrame(
            np.vstack(R_summary), 
            index=self.pmuds_kwargs.keys(), 
            columns=pd.MultiIndex.from_tuples(multi_index)
        )
        
