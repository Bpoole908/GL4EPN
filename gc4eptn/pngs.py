from pdb import set_trace
from functools import partial

import numpy as np
import pandas as pd
import dataframe_image as dfi
from matplotlib.transforms import Bbox
from IPython.display import display

from gc4eptn.ggm import single_ggm_exp
from gc4eptn.utils.plotting import plot_recovered

    
class PowerNetworkGraphScore():
    _score_types = ['current', 'mpl', 'apfd', 'sum', 'and']
    def __init__(self, pmu_data, verbose: bool = False):
        self.data = pmu_data
        self.nodes = pmu_data.nodes
        self.verbose = verbose
        
    def __call__(self, A_hat):
        # NOTE: This assumes that nodes are ordered from least to greatest node number
        # For example, BUS5 comes before BUS6 in the node ordering
        A_hat = pd.DataFrame(A_hat, index=self.nodes, columns=self.nodes)
        self.I_scores = A_hat.copy().astype(object)
        self.I_scores[:] = np.nan
        self.P_apfd_scores = A_hat.copy().astype(object)
        self.P_apfd_scores[:] = np.nan
        self.P_mpl_scores = A_hat.copy().astype(object)
        self.P_mpl_scores[:] = np.nan
            
        # loop over every node in original but only follow connections given by A_hat
        for se_node in self.nodes:  
            # Get the predicted edges for current node
            pred_connections_adj_row = A_hat[se_node]
            n_pred_connections = pred_connections_adj_row.sum()
            # Get the predicted connected nodes for current node
            pred_connections = self.nodes[pred_connections_adj_row > 0]

            se_I_unk = self.data.get_node_value(se_node, 'unknown')
            se_I_outs = np.hstack(self.data.get_node_value(se_node, 'outs'))
            se_I_ins = np.hstack(self.data.get_node_value(se_node, 'ins')) 
            se_V =  np.hstack(self.data.get_node_value(se_node, 'voltage'))
          # If the number of outs is equal to zero, we continue as there are no outs to check
            if len(se_I_outs) == 0:
                print(f"Skipping node check for {se_node} as it has no outs.") if self.verbose else None
                self.I_scores.loc[se_node, pred_connections] = np.inf
                self.P_apfd_scores.loc[se_node, pred_connections] = np.inf
                self.P_mpl_scores.loc[se_node, pred_connections] = np.inf
                continue
            # NOTE: Not needed, If the number of original outs is greater than the number of edges, 
            #       node has invalid connections.
            if len(se_I_outs) > n_pred_connections and self.verbose:
                print(f"Number of original 'outs' ({len(se_I_outs)}) > predicted number of edges ({n_pred_connections}) for {se_node}")
                # continue

            # NOTE: Not needed, If the number of original ins is greater than the number of edges, 
            #       node has  invalid connections
            if len(se_I_ins) > n_pred_connections and self.verbose:
                print(f"Number of original 'ins' ({len(se_I_ins)}) > predicted number of edges ({n_pred_connections}) for {se_node}")
                # continue
                  
            self.score_connections(
                se_node=se_node, 
                se_I_outs=se_I_outs, 
                se_V=se_V, 
                pred_connections=pred_connections
            )
        
    def score_connections(self, se_node, se_I_outs, se_V, pred_connections):
        for pred_re_node in pred_connections:
            re_I_unk = self.data.get_node_value(se_node, 'unknown')
            re_I_ins = np.hstack(self.data.get_node_value(pred_re_node, 'ins'))
            re_V =  np.hstack(self.data.get_node_value(pred_re_node, 'voltage'))
            # If there are no ins for potentially connected node, then this edge is a false edge
            if len(re_I_ins) == 0:
                print(f"Skipping node check from {se_node} to {pred_re_node} as node {pred_re_node} has no ins.") if self.verbose else None
                self.I_scores.loc[se_node, pred_re_node] = np.inf
                self.P_apfd_scores.loc[se_node, pred_re_node] = np.inf
                self.P_mpl_scores.loc[se_node, pred_re_node] = np.inf
                continue

            I_scores, P_apfd_scores, P_mpl_scores = self.compute_flow_difference(
                se_I_outs=se_I_outs, 
                se_V=se_V,
                re_I_ins=re_I_ins, 
                re_V=re_V
            )
            self.I_scores.loc[se_node, pred_re_node] = I_scores
            self.P_apfd_scores.loc[se_node, pred_re_node] = P_apfd_scores
            self.P_mpl_scores.loc[se_node, pred_re_node] = P_mpl_scores
            
            # DEBUG
            if self.verbose:
                print(re_I_ins)
                print(f"{se_node} -> {pred_re_node}", I_scores)
    
    def compute_flow_difference(self, se_I_outs, se_V, re_I_ins, re_V):
        se_I_outs_mag_col = self.data.filter(self.data.df[se_I_outs.flatten()], self.data.magnitude_tags)
        se_I_outs_mag = self.data.df[se_I_outs_mag_col].values.T
        se_I_outs_ang_col = self.data.filter(self.data.df[se_I_outs.flatten()], self.data.angle_tags)
        se_I_outs_ang = self.data.df[se_I_outs_ang_col].values.T
        se_V_mag_col = self.data.filter(self.data.df[se_V.flatten()], self.data.magnitude_tags)
        se_V_mag = self.data.df[se_V_mag_col].values.T # only ever 1 sending node voltage
        se_V_ang_col = self.data.filter(self.data.df[se_V.flatten()], self.data.angle_tags)
        se_V_ang = self.data.df[se_V_ang_col].values.T # only ever 1 sending node voltage
        
        re_I_ins_mag_col = self.data.filter(self.data.df[re_I_ins.flatten()], self.data.magnitude_tags)
        re_I_ins_mag = self.data.df[re_I_ins_mag_col].values.T
        re_I_ins_ang_col = self.data.filter(self.data.df[re_I_ins.flatten()], self.data.angle_tags)
        re_I_ins_ang = self.data.df[re_I_ins_ang_col].values.T
        re_V_mag_col = self.data.filter(self.data.df[re_V.flatten()], self.data.magnitude_tags)
        re_V_mag = self.data.df[re_V_mag_col].values.T
        re_V_ang_col = self.data.filter(self.data.df[re_V.flatten()], self.data.angle_tags)
        re_V_ang = self.data.df[re_V_ang_col].values.T

        I_scores = np.empty([len(se_I_outs_mag_col), len(re_I_ins_mag_col)])
        P_apfd_scores = np.empty([len(se_I_outs_mag_col), len(re_I_ins_mag_col)])
        P_mpl_scores = np.empty([len(se_I_outs_mag_col), len(re_I_ins_mag_col)])
        # Loop over each sending out one at a time to compute scores with ALL receiving ins at once
        for i, (se_I_out_mag, se_I_out_ang) in enumerate(zip(se_I_outs_mag, se_I_outs_ang)):
            se_I_out_mag = se_I_out_mag.reshape(1, -1)
            se_I_out_ang = se_I_out_ang.reshape(1, -1)
            
            # Mean current flow difference
            acfd = np.mean(np.abs( np.abs(se_I_out_mag) - np.abs(re_I_ins_mag) ), axis=1)
            I_scores[i] = acfd
            
            # Absolute power flow difference
            se_pf = se_V_mag * se_I_out_mag * np.cos(se_V_ang - se_I_out_ang)
            re_pf = re_V_mag * re_I_ins_mag * np.cos(re_V_ang - re_I_ins_ang)
            apfd = np.abs(np.abs(se_pf) - np.abs(re_pf))
            P_apfd_scores[i] = np.mean(apfd, axis=1)
            
            # Mean power loss
            se_I = self.phasor(mag=se_I_out_mag, ang=se_I_out_ang)
            se_V = self.phasor(mag=se_V_mag, ang=se_V_ang)
            re_I = self.phasor(mag=re_I_ins_mag, ang=re_I_ins_ang)
            re_V = self.phasor(mag=re_V_mag, ang=re_V_ang)
            I_se_re = (se_I + re_I) / 2
            V_se_re = se_V - re_V
            mpl = np.abs( np.real( V_se_re * np.conj(I_se_re) ) )
            P_mpl_scores[i] = np.mean(mpl, axis=1)
          
        return I_scores, P_apfd_scores, P_mpl_scores

    def phasor(self, mag, ang):
        return (mag * np.cos(ang)) + (mag * np.sin(ang) * 1j)
    
    def get_refined_adjacency(self, use_score='sum', min_threshold=0.2, axis=1):
        assert use_score in self._score_types
        if use_score == 'current':
            return self._score_to_adj(self.I_scores, min_threshold=min_threshold, axis=axis).astype(int)
        elif use_score == 'apfd':
            return self._score_to_adj(self.P_apfd_scores, min_threshold=min_threshold, axis=axis).astype(int)
        elif use_score == 'mpl':
            return self._score_to_adj(self.P_mpl_scores, min_threshold=min_threshold, axis=axis).astype(int)
        else:
            return self._combine_scores(combination_type=use_score, min_threshold=min_threshold, axis=axis)
        
    def _score_to_adj(self, scores, min_threshold, axis=1):
        scores = scores.map(partial(self._min, axis=axis))
        return scores.transform(
            partial(self._top_min_scores, axis=axis, min_threshold=min_threshold), 
            axis=axis
        )
        
    def _min(self, x, axis=None):
        if isinstance(x, np.ndarray): 
            if len(x.shape) == 1:
                x = x.reshape(-1, 1)
        else:
            return x
        return np.min(x, axis=axis)
        
    def _top_min_scores(self, x: pd.Series, min_threshold: float, axis: int):
        n_flows = self.data.node_info[x.name]['n_outs' if axis == 1 else 'n_ins']
        adj = np.zeros(len(x), dtype=int)
        flattened_x = x.explode()

        top_scores_idx = np.hstack(flattened_x).argsort()[:n_flows]
        top_scores = flattened_x.iloc[top_scores_idx]
        # Loop over top scores and validate them
        for v, idx in zip(top_scores, top_scores.index):
            if v == np.inf or np.isnan(v) or v >= min_threshold:
                continue
            iloc = x.index.get_loc(idx)
            adj[iloc] += 1 
        return adj
    
    def _combine_scores(self, combination_type, min_threshold, axis):
        if combination_type == 'and':
            I_adj = self._score_to_adj(self.I_scores, min_threshold=min_threshold, axis=axis)
            P_apfd_adj = self._score_to_adj(self.P_apfd_scores, min_threshold=min_threshold, axis=axis)
            P_mpl_adj = self._score_to_adj(self.P_mpl_scores, min_threshold=min_threshold, axis=axis)
            return pd.DataFrame(
                np.minimum.reduce([I_adj, P_apfd_adj, P_mpl_adj]).astype(int),
                index=self.I_scores.index,
                columns=self.I_scores.columns,
            )
        elif combination_type == 'sum':
            return self._score_to_adj(
                self.I_scores + self.P_apfd_scores + self.P_mpl_scores, 
                min_threshold=min_threshold, 
                axis=axis
            )
    
    def _color_gradient(self, df):
            return (
                df.replace(np.inf, -1)
                .style.background_gradient(axis=None, cmap='BuGn', low=0)
                .map(lambda x: 'background-color: black; color: white' if x < 0 else '')
            )
        
    def get_styled_simple_graph_score(self, use_score='sum', dec=5):
        assert use_score in self._score_types
        if use_score == 'current':
            scores = self.I_scores.map(self._min).round(dec)
        elif use_score == 'apfd':
            scores = self.P_apfd_scores.map(self._min).round(dec)
        elif use_score == 'mpl':
            scores = self.P_mpl_scores.map(self._min).round(dec)
        elif use_score == 'sum':
            scores = (self.I_scores+self.P_apfd_scores+self.P_mpl_scores).map(self._min).round(dec)
        elif use_score == 'and':
            msg = "The score 'and' does not have a graph score and can only be used when generating the adjacency matrix."
            raise ValueError(msg)

        scores_style = self._color_gradient(scores)
        return scores_style
    

def run_ggm_pngs_exp(
    net_pmuds, 
    flow_pmuds, 
    R,
    n_samples,
    penalty,
    pngs_score_type,
    pngs_min_threshold, 
    select_best_metric_fn, 
    select_best_metric_type, 
    ggmncv_kwargs,
    save_path,
    ggm_metrics_fn,
    refined_metrics_fn,
    include_ggm_result_metrics=None,
    verbose: bool = False,
    disable_plotting: bool = False,
):
    save_path_pen = save_path/penalty
    include_ggm_result_metrics = include_ggm_result_metrics or []
    A_hat, _, info, (fig, axs), results = single_ggm_exp(
        R=R,
        n_samples=n_samples,
        penalty=penalty,
        A=net_pmuds.true_network_graph,
        positions=net_pmuds.graph_positions,
        labels=net_pmuds.labels,
        tau=None,
        annot=True,
        save_dir_path=save_path,
        ggmncv_kwargs=ggmncv_kwargs,
        metric_fn=select_best_metric_fn,
        select_metric=select_best_metric_type,
        verbose=verbose,
        disable_plotting=disable_plotting
    )
    ggm_metrics = ggm_metrics_fn(
        A=net_pmuds.true_network_graph, 
        A_hat=A_hat,
        directed=False,
    )
    params = info['params']
    for m in include_ggm_result_metrics:
        ggm_metrics[m] = info['metrics'].get(m, np.nan)
        
    refined_A_hat, pngs = run_pngs(
        flow_pmuds,
        A_hat,
        score_type=pngs_score_type, 
        min_threshold=pngs_min_threshold,
        verbose=verbose,
        save_path=save_path_pen
    )
    refined_metrics = refined_metrics_fn(
        A=flow_pmuds.true_flow_graph, 
        A_hat=refined_A_hat.values,
        directed=True,
    )
    
    extent = axs[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    axs[0].set_title('')
    fig.savefig(save_path_pen/'estimated-graph.pdf', bbox_inches=extent)
    extent = axs[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    axs[1].set_title('')
    fig.savefig(save_path_pen/'estimated-precision.pdf', bbox_inches=Bbox.from_extents(4.7, -0.1, 9.8, 4.5))

    fig, axs = plot_recovered(
        A=refined_A_hat.values,
        theta=refined_A_hat.values, 
        A_diff=A_hat,
        labels=net_pmuds.labels,
        positions=net_pmuds.graph_positions,
        annot=True,
        cbar=True,
        directed=False,
        title="Graph Difference",
        sub_titles=['Graph', 'Adjacency Matrix'],
    )
    extent = axs[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(save_path_pen/'diff-estimated-graph.pdf', bbox_inches=extent)
    
    return ggm_metrics, refined_metrics, params, pngs 


def save_scores(pngs, axis, save_path, verbose=False):
    save_path.mkdir(parents=True, exist_ok=True)
    
    def save_refined_adj(use_score, score_name, ajd_name):
        if score_name is not None:
            scores = pngs.get_styled_simple_graph_score(use_score=use_score)
            dfi.export(scores, save_path/score_name) if save_path is not None else ''
        
        A_hat = pngs.get_refined_adjacency(use_score=use_score, axis=axis)
        A_hat_style = A_hat.style.map(lambda x: f"background-color: green;" if x != 0 else None)
        dfi.export(A_hat_style, save_path/ajd_name)
        if verbose:
            if score_name is not None: display(scores)
            display(A_hat_style)

    save_refined_adj(use_score='current', score_name=f"I-scores.pdf", ajd_name=f"I-adj.pdf")
    save_refined_adj(use_score='apfd', score_name=f"P-apfd-scores.pdf", ajd_name=f"P-apfd-adj.pdf")
    save_refined_adj(use_score='mpl',score_name=f"P-mpl-scores.pdf",  ajd_name=f"P-mpl-adj.pdf")
    save_refined_adj(use_score='sum',score_name=f"IP-sum-scores.pdf",  ajd_name=f"IP-sum-adj.pdf")
    save_refined_adj(use_score='and', score_name=None, ajd_name=f"IP-and-adj.pdf")


def run_pngs(
    flow_pmuds, 
    A_hat,
    min_threshold,
    score_type='sum', 
    axis=1, 
    verbose=False, 
    save_path=None
):
    pngs = PowerNetworkGraphScore(flow_pmuds, verbose=False)
    pngs(A_hat)
    
    save_scores(pngs, axis=1, save_path=save_path/'outs', verbose=verbose)
    save_scores(pngs, axis=0, save_path=save_path/'ins', verbose=verbose)
    refined_A_hat = pngs.get_refined_adjacency(
        use_score=score_type, 
        min_threshold=min_threshold, 
        axis=axis
    )

    arcs = _get_arcs(flow_pmuds, refined_A_hat.values)
    fig, axs = plot_recovered(
        A=refined_A_hat.values,
        theta=refined_A_hat.values, 
        labels=flow_pmuds.labels,
        positions=flow_pmuds.graph_positions,
        annot=True,
        cbar=True,
        directed=True,
        arcs=arcs,
        title="Refined Graph",
        sub_titles=['Graph', 'Adjacency Matrix'],
        save_path=save_path/'refined-graph-precision.pdf' if save_path is not None else None
    )

    if save_path is not None:
        extent = axs[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        axs[0].set_title('')
        fig.savefig(save_path/'refined-graph.pdf', bbox_inches=extent) 
        extent = axs[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        axs[1].set_title('')
        fig.savefig(save_path/'refined-precision.pdf', bbox_inches=Bbox.from_extents(4.7, -0.1, 9.8, 4.5))
    
    return refined_A_hat, pngs


def _get_arcs(flow_pmuds, A_hat):
    arcs = []
    diff = (flow_pmuds.true_flow_graph - A_hat).astype(int)
    if diff.sum() == 0:
        return flow_pmuds.directed_arcs

    row_iter = 0
    for i, r in enumerate(A_hat):
        rad = -.20
        for j, c in enumerate(r):
            edges = int(flow_pmuds.true_flow_graph[i, j])
            # print("True edges", flow_pmuds.true_flow_graph[i, j], A_hat[i, j],  diff[i, j])
            if flow_pmuds.true_flow_graph[i, j]  > 0:
                if A_hat[i, j] > 0:
                    # Add correct arc when ground truth and A_hat match
                    arcs.extend(flow_pmuds.directed_arcs[row_iter:row_iter+edges])
                row_iter += edges
            if diff[i, j] < 0:
                # Add arc for false edges
                false_edges = np.abs(diff[i, j])
                arcs.extend([f"arc3,rad={rad}"]*false_edges)
                rad -= .20
                
    assert len(arcs) == A_hat.sum()
    return arcs