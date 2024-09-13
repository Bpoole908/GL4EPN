from pdb import set_trace
from pathlib import Path

import itertools as it
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import ConnectionStyle
from matplotlib.transforms import Bbox
from gglasso.helper.model_selection import thresholding

from gc4eptn.utils.utils import to_adjacency
from gc4eptn.utils.kernels import weighted_kernel
def draw_directed_graph(
    A,
    positions, 
    labels,
    arrows=True,
    node_size=800,
    font_size=24,
    line_width=4,
    arrowstyle="->",
    arrowsize=30,
    save_path=None,
):
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    fig, ax = plt.subplots(1,1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    nx.draw_networkx(
        G,
        positions,
        labels=labels,
        arrows=arrows,
        arrowstyle=arrowstyle,
        arrowsize=arrowsize,
        font_color = 'white', 
        node_color = "black", 
        edge_color = "black", 
        font_size=font_size,
        node_size=node_size,
        width=line_width, 
        connectionstyle='arc3, rad = 0.1',
        ax=ax,

    )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path))

def plot_adjacency(
    A,
    labels,
    title='',
    annot=True,
    number_fmt='.4f',
    vmin=None,
    vmax=None,
    save_path=None,
):
    fig, ax = plt.subplots(1,1)
    ax.set_title(title) 
    sns.heatmap(
            A, 
            cmap="coolwarm", 
            vmin=vmin, 
            vmax=vmax, 
            linewidth=.5, 
            square=True,
            cbar=True,
            annot=annot,
            fmt=number_fmt,
            annot_kws={"fontsize":6},
            xticklabels=labels, 
            yticklabels=labels,
            ax=ax
        )
        
    if save_path is not None:
        fig.savefig(str(save_path))


def plot_ground_truth(
    A, 
    positions, 
    labels, 
    node_size=800, 
    font_size=24, 
    line_width=4,
    annot=True,
    cbar=True,
    save_path=None,
    directed=False,
    arcs=None,
):
    assert labels is None or isinstance(labels, dict)

    ticks = np.arange(len(A)) if labels is None else labels.values()
    if not annot:
        ticks = False
        
    fig, axs = plt.subplots(1,2, figsize=(10,5))
    fig.suptitle("Ground Truth")
    axs[0].axis('off')
    axs[0].set_title("True graph")
    draw_network(
        A,
        positions=positions,
        labels=labels,
        ax=axs[0],
        node_size=node_size,
        font_size=font_size,
        line_width=line_width,
        directed=directed,
        arcs=arcs,
    )
    axs[1].set_title("True Adjacency matrix")
    sns.heatmap(
        A, 
        cmap="coolwarm", 
        vmin=-0.5, 
        vmax=0.5, 
        linewidth=.5, 
        square=True,
        cbar=cbar,
        annot=annot,
        annot_kws={"fontsize":8},
        fmt='.1g',
        xticklabels=ticks, 
        yticklabels=ticks, 
        ax=axs[1],
    )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path))
    
    return fig, axs


# TODO: Move connectionstyle out of this function
# def draw_network(
#     A,
#     positions=None,
#     labels=None,
#     ax=None,
#     node_size=800,
#     font_size=24,
#     line_width=4,
#     directed=False,
#     from_numpy_kwargs=None
# ):
#     from_numpy_kwargs = from_numpy_kwargs or {'parallel_edges':True}
#     # draw_kwargs = dict(connectionstyle=[f"arc3,rad={r}" for r in it.accumulate([-.3, .3])])
#     draw_kwargs = dict(connectionstyle=[
#         f"arc3,rad={0}",
#         f"arc3,rad={.1}",
#     ])
#     if ax is None:
#         _, ax = plt.subplots()
        
#     if directed:
#         draw_kwargs = dict(
#             arrows=True,
#             arrowstyle="->",
#             arrowsize=30,
#             **draw_kwargs
#         )
#         from_numpy_kwargs['create_using'] = nx.MultiDiGraph
#         G = nx.from_numpy_array(A.astype(int), **from_numpy_kwargs)
#     else:
#         from_numpy_kwargs['create_using'] = nx.MultiGraph
#         G = nx.from_numpy_array(A.astype(int), **from_numpy_kwargs)

#     positions = positions or nx.circular_layout(G)
#     print(positions)
#     nx.draw_networkx(
#         G, 
#         pos=positions, 
#         node_size=node_size,
#         node_color="black", 
#         edge_color="black", 
#         labels=labels,
#         font_size=font_size, 
#         font_color='white', 
#         with_labels=True,
#         width=line_width, 
#         ax=ax,
#         **draw_kwargs
#     )

def draw_network(
    A,
    positions=None,
    labels=None,
    ax=None,
    node_size=800,
    font_size=24,
    line_width=4,
    directed=False,
    arcs=None,
    from_numpy_kwargs=None
):
    from_numpy_kwargs = from_numpy_kwargs or {'parallel_edges':True}
    # draw_kwargs = dict(connectionstyle=[f"arc3,rad={r}" for r in it.accumulate([-.3, .3])])
    draw_kwargs = {}
    if ax is None:
        _, ax = plt.subplots()
        
    if directed:
        draw_kwargs = dict(
            arrows=True,
            arrowstyle="->",
            arrowsize=30,
        )
        from_numpy_kwargs['create_using'] = nx.MultiDiGraph
        G = nx.from_numpy_array(A.astype(int), **from_numpy_kwargs)
    else:
        from_numpy_kwargs['create_using'] = nx.MultiGraph
        G = nx.from_numpy_array(A.astype(int), **from_numpy_kwargs)

    positions = positions or nx.circular_layout(G)
    nx.draw_networkx_nodes(G, positions, node_size=node_size, node_color="black", ax=ax)
    if arcs is None:
        arcs =["arc3,rad=0"]*G.number_of_edges()

    for i, edge in enumerate(G.edges(data=True)):
        arc = nx.draw_networkx_edges(
            G, 
            positions, 
            edgelist=[(edge[0],edge[1])], 
            edge_color="black",
            width=line_width,
            connectionstyle=arcs[i],
            ax=ax,
            **draw_kwargs
        )
    nx.draw_networkx_labels(G, positions, labels=labels, font_color='white', font_size=font_size, ax=ax)


def draw_network_diff(
    original, 
    new,
    positions=None,
    labels=None,
    ax=None,
    node_size=800,
    font_size=24,
    line_width=4,
    directed=False,
):
    from_numpy_kwargs = {}
    draw_kwargs = {}
    if ax is None:
        _, ax = plt.subplots()
        
    if directed:
        draw_kwargs = dict(
            arrows=True,
            arrowstyle="->",
            arrowsize=30,
        )
        from_numpy_kwargs = dict(create_using=nx.DiGraph)

    G = nx.from_numpy_array(original, **from_numpy_kwargs)
    positions = positions or nx.circular_layout(G)

    diff = np.abs(original - new)
    G_new = nx.from_numpy_array(new, **from_numpy_kwargs)
    G_diff = nx.from_numpy_array(diff, **from_numpy_kwargs)

    nx.draw_networkx_nodes(
        G, 
        ax=ax,
        pos=positions,
        node_size=node_size,
        node_color="black", 
    )
    nx.draw_networkx_labels(
        G,
        ax=ax,
        pos=positions,
        labels=labels,
        font_size=font_size,
        font_color='white',
    )
    nx.draw_networkx_edges(
        G_new,
        ax=ax,
        pos=positions,  
        width=line_width,
        edge_color="black",
        style='-',
        connectionstyle='arc3',
        **draw_kwargs
    )
    nx.draw_networkx_edges(
        G_diff,
        ax=ax,
        pos=positions,  
        width=line_width,
        edge_color="black",
        style=':',
        connectionstyle='arc3',
        **draw_kwargs
    )
     
def plot_recovered(
    A,
    theta, 
    positions,
    arcs=None,
    A_diff=None, 
    labels=None,
    title='',
    sub_titles=['', ''],
    node_size=800,
    font_size=24,
    line_width=4,
    directed=False,
    annot=True,
    cbar=True,
    number_fmt='.4f',
    vmin=-0.5,
    vmax=0.5,
    annot_kws=None,
    save_path=None,
):
    assert labels is None or isinstance(labels, dict)
    annot_kws = {"fontsize":8} if annot_kws is None else annot_kws

    # Build figure and ax
    ticks = np.arange(len(A)) if labels is None else labels.values()
    if not annot:
        ticks = False
        
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=80)
    fig.suptitle(title)
    axs[0].axis('off')
    axs[0].set_title(sub_titles[0])
    if A_diff is not None:
        draw_network_diff(
            A_diff,
            A,
            positions=positions,
            labels=labels,
            ax=axs[0],
            node_size=node_size,
            font_size=font_size,
            line_width=line_width,
            directed=directed,
        )
    else:
        draw_network(
            A,
            positions=positions,
            labels=labels,
            ax=axs[0],
            node_size=node_size,
            font_size=font_size,
            line_width=line_width,
            directed=directed,
            arcs=arcs
        )
    axs[1].set_title(sub_titles[1])
    sns.heatmap(
        theta, 
        cmap="coolwarm", 
        vmin=vmin, 
        vmax=vmax, 
        linewidth=.5, 
        square=True,
        cbar=cbar,
        annot=annot,
        fmt=number_fmt,
        annot_kws=annot_kws,
        xticklabels=ticks, 
        yticklabels=ticks,
        ax=axs[1]
    )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path))
        
    return fig, axs

def plot_matrices(
    theta, 
    tau, 
    labels=None,
    number_fmt='.4f', 
    title='', 
    save_path=None,
    annot=True,
): 
    if labels is None:
        labels = np.arange(len(A))
    if not annot:
        labels = False
        
    fig, axs = plt.subplots(1,3, figsize=(15,8))
    fig.suptitle(title)

    axs[0].set_title("Recovered matrix")
    sns.heatmap(
        theta, 
        cmap="coolwarm", 
        vmin=-0.5, 
        vmax=0.5, 
        linewidth=.5, 
        square=True,
        cbar=False,
        cbar_ax=axs[-1],
        annot=annot,
        fmt=number_fmt,
        annot_kws={"fontsize":8},
        xticklabels=labels, 
        yticklabels=labels,
        ax=axs[0]
    )

    axs[1].set_title(f"Thresholded matrix ({np.round(tau,5)})")
    theta_tau = thresholding(theta, tau=tau) if tau is not None else theta
    sns.heatmap(
        theta_tau, 
        cmap="coolwarm", 
        vmin=-0.5, 
        vmax=0.5, 
        linewidth=.5, 
        square=True,
        cbar=False,
        cbar_ax=axs[-1],
        annot=annot,
        fmt=number_fmt,
        annot_kws={"fontsize":8},
        xticklabels=labels, 
        yticklabels=labels, 
        ax=axs[1]
    )

    axs[2].set_title("Recovered adjacency matrix")
    sns.heatmap(
        to_adjacency(theta_tau), 
        cmap="coolwarm", 
        vmin=-0.5, 
        vmax=0.5, 
        linewidth=.5, 
        square=True,
        cbar=False,
        annot=annot,
        fmt=number_fmt,
        annot_kws={"fontsize":8},
        xticklabels=labels, 
        yticklabels=labels,
        ax=axs[2]
    )

    if save_path is not None:
        fig.savefig(str(save_path))
        
def plot_pmu_weighted_kernel_matrix(
    pmuds, 
    kernel=None, 
    taus=[0.5, 0.5],
    fmt='.3f',
    annot_kws=None,
    save_dir=None
):
    annot_kws = annot_kws or {"fontsize":10}
    save_dir = Path(save_dir) if save_dir is not None else None
    kernel = np.corrcoef if kernel is None else kernel

    graph_df = pmuds.graph_df
    tick_labels = pmuds.labels.values()
    
    R, dfs = weighted_kernel(
        graph_df, 
        kernel=kernel,
        columns=['mag', 'ang'], 
        weights=taus, 
        return_all=True
    )
    ma_R = dfs['mag']
    an_R = dfs['ang']
    print(f"Weighted cond: {np.linalg.cond(R)}")
    ax = sns.heatmap(
        R, 
        cmap="coolwarm", 
        linewidth=.5, 
        square=True,
        cbar=True,
        annot=True,
        fmt=fmt,
        annot_kws=annot_kws,
        xticklabels=tick_labels, 
        yticklabels=tick_labels,  

    )
    plt.show()
    
    print(f"Magnitude cond: {np.linalg.cond(ma_R)}")
    ma_ax = sns.heatmap(
        ma_R, 
        cmap="coolwarm", 
        linewidth=.5, 
        square=True,
        cbar=True,
        annot=True,
        fmt=fmt,
        annot_kws=annot_kws,
        xticklabels=tick_labels, 
        yticklabels=tick_labels, 
    )
    plt.show()
    
    print(f"Angle cond: {np.linalg.cond(an_R)}")
    an_ax = sns.heatmap(
        an_R, 
        cmap="coolwarm", 
        linewidth=.5, 
        square=True,
        cbar=True,
        annot=True,
        fmt=fmt,
        annot_kws=annot_kws,
        xticklabels=tick_labels, 
        yticklabels=tick_labels, 
    )
    plt.show()
    
    if save_dir is not None:
        an_ax.figure.savefig(str(save_dir/'ang-corr-mat.pdf'))
        ma_ax.figure.savefig(str(save_dir/'mag-corr-mat.pdf'))
        ax.figure.savefig(str(save_dir/'corr-mat.pdf'))

    return R, ma_R, an_R


def plot_heatmap(R, xticklabels, yticklabels, fmt, annot_kws, save_path=None):
    ax = sns.heatmap(
        R, 
        cmap="coolwarm", 
        linewidth=.5, 
        square=True,
        cbar=True,
        annot=True,
        annot_kws=annot_kws,
        fmt=fmt,
        xticklabels=xticklabels, 
        yticklabels=yticklabels,  
    )
    if save_path:
        ax.figure.savefig(save_path)
        
    return ax


def plot_eptn_ground_truths(save_path, net_pmuds, flow_pmuds, annot_kws=None, verbose=False):
    sdp_gt = save_path/'ground-truth'
    sdp_gt.mkdir(parents=True, exist_ok=True)
    
    # Plot and save ground truth netowrk (undirected) graph used for GGMs/GSP (net_pmuds)
    fig, axs = plot_recovered(
        A=net_pmuds.true_network_graph,
        theta=net_pmuds.true_network_graph,
        labels=net_pmuds.labels,
        positions=net_pmuds.graph_positions,
        annot=True,
        cbar=True,
        number_fmt='.1f',
        annot_kws=annot_kws,
        title="Ground Truth",
        sub_titles=['Graph', 'Adjacency Matrix'],
        save_path=sdp_gt/'graph-adj.pdf',
        arcs=net_pmuds.undirected_arcs
    )
    extent = axs[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(sdp_gt/'graph.pdf', bbox_inches=extent)
    extent = axs[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    axs[1].set_title('')
    fig.savefig(sdp_gt/'adj.pdf', bbox_inches=Bbox.from_extents(4.7, -0.1, 9.8, 4.5))
    
    # Plot and save ground truth flow (directed) graph used for PNGR (flow_pmuds)
    fig, axs = plot_recovered(
        A=flow_pmuds.true_flow_graph,
        theta=flow_pmuds.true_flow_graph,
        labels=flow_pmuds.labels,
        positions=flow_pmuds.graph_positions,
        annot=True,
        cbar=True,
        directed=True,
        number_fmt='.1f',
        annot_kws=annot_kws,
        title="Ground Truth",
        sub_titles=['Graph', 'Adjacency Matrix'],
        save_path=sdp_gt/'directed-graph-adj.pdf',
        arcs=flow_pmuds.directed_arcs
    )
    extent = axs[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # bbox_inches=extent if drop_parallel_currents else Bbox.from_extents(0.752, 0.15, 4.23, 3.1)
    fig.savefig(sdp_gt/'directed-graph.pdf', bbox_inches=extent)
    extent = axs[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    axs[1].set_title('')
    fig.savefig(sdp_gt/'directed-adj.pdf', bbox_inches=Bbox.from_extents(4.7, -0.1, 9.8, 4.5))

    if verbose:
        plt.show()
    plt.clf()
    plt.close('all')


def R_mag_ang_plots(
    R_func, 
    pmuds_class, 
    load_kwargs, 
    build_kwargs, 
    save_path, 
    filename, 
    fmt='.2f',
    annot_kws=None,
    verbose=False
):
    
    pmuds = pmuds_class(**load_kwargs).build_graph_data(**build_kwargs)
    R = R_func(pmuds.graph_df)
    plot_R(pmuds, R, save_path=save_path, filename=filename, fmt=fmt, annot_kws=annot_kws, verbose=verbose)
    
    pmuds = pmuds_class(drop_angle=True, **load_kwargs).build_graph_data(**build_kwargs)
    R_mag = R_func(pmuds.graph_df)
    plot_R(pmuds, R_mag, save_path=save_path, filename=f'{filename}-mag', fmt=fmt, annot_kws=annot_kws, verbose=verbose)
    
    pmuds = pmuds_class(drop_magnitude=True, **load_kwargs).build_graph_data(**build_kwargs)
    R_ang = R_func(pmuds.graph_df)
    plot_R(pmuds, R_ang,  save_path=save_path, filename=f'{filename}-ang', fmt=fmt, annot_kws=annot_kws, verbose=verbose)

    
def plot_R(pmuds, R, save_path, filename=None, fmt='.2f', annot_kws=None, verbose=False):
    plot_heatmap(
        R, 
        xticklabels=pmuds.labels.values(), 
        yticklabels=pmuds.labels.values(), 
        fmt=fmt, 
        annot_kws=annot_kws,
        save_path=save_path/f"{filename}.pdf" if filename is not None else None
    )
    if verbose: plt.show()
    plt.clf()
    plt.clf()
    plt.close('all')