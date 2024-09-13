from pdb import set_trace

import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

from gc4eptn.utils.metrics import hamming_distance
from gc4eptn.utils.plotting import plot_recovered
from gc4eptn.utils.utils import to_adjacency, thresholding, filter_function_kwargs

solvers.options['show_progress'] = False


def gl_sig_model(inp_signal, max_iter, alpha, beta):
    """
    Returns Output Signal Y, Graph Laplacian L
    """
    Y = inp_signal.T
    num_vertices = inp_signal.shape[1]
    M_mat, P_mat, A_mat, b_mat, G_mat, h_mat = create_static_matrices_for_L_opt(num_vertices, beta)

    # M_c = matrix(M_mat)
    P_c = matrix(P_mat)
    A_c = matrix(A_mat)
    b_c = matrix(b_mat)
    G_c = matrix(G_mat)
    h_c = matrix(h_mat)

    curr_cost = np.linalg.norm(np.ones((num_vertices, num_vertices)), 'fro')
    q_mat = alpha * np.dot(np.ravel(np.dot(Y, Y.T)), M_mat)
    for it in range(max_iter):
        # Update L
        prev_cost = curr_cost
        q_c = matrix(q_mat)
        sol = solvers.qp(P_c, q_c, G_c, h_c, A_c, b_c)
        l_vech = np.array(sol['x'])
        l_vec = np.dot(M_mat, l_vech)
        L = l_vec.reshape(num_vertices, num_vertices)
        # Assert L is correctly learnt.
        # assert L.trace() == num_vertices
        assert np.allclose(L.trace(), num_vertices)
        L_diff = L - np.diag(np.diag(L))
        assert np.all(L_diff <= 0), L_diff[L_diff > 0]
        assert np.allclose(np.dot(L, np.ones(num_vertices)), np.zeros(num_vertices))

        # Update Y
        R = np.linalg.cholesky(np.eye(num_vertices) + alpha * L)
        Y = np.linalg.solve(R, np.linalg.solve(R.T, inp_signal.T))
        # Y = np.dot(np.linalg.inv(np.eye(num_vertices) + alpha * L), inp_signal.T)
 
        curr_cost = (np.linalg.norm(inp_signal.T - Y, 'fro')**2 +
                     alpha * np.dot(np.dot(Y.T, L), Y).trace() +
                     beta * np.linalg.norm(L, 'fro')**2)
        q_mat = alpha * np.dot(np.ravel(np.dot(Y, Y.T)), M_mat)

        calc_cost = (0.5 * np.dot(np.dot(l_vech.T, P_mat), l_vech).squeeze() +
                     np.dot(q_mat, l_vech).squeeze() + np.linalg.norm(inp_signal.T - Y, 'fro')**2)

        assert np.allclose(curr_cost, calc_cost)

        if np.abs(curr_cost - prev_cost) < 1e-4:
            break

    return L, Y


def create_static_matrices_for_L_opt(num_vertices, beta):
    # Static matrices are those independent of Y
    #
    M_mat = create_dup_matrix(num_vertices)
    P_mat = 2 * beta * np.dot(M_mat.T, M_mat)
    A_mat = create_A_mat(num_vertices)
    b_mat = create_b_mat(num_vertices)
    G_mat = create_G_mat(num_vertices)
    h_mat = np.zeros(G_mat.shape[0])
    return M_mat, P_mat, A_mat, b_mat, G_mat, h_mat


def create_dup_matrix(num_vertices):
    M_mat = np.zeros((num_vertices**2, num_vertices*(num_vertices + 1)//2))
    # tmp_mat = np.arange(num_vertices**2).reshape(num_vertices, num_vertices)
    for j in range(1, num_vertices+1):
        for i in range(j, num_vertices+1):
            u_vec = get_u_vec(i, j, num_vertices)
            Tij = get_T_mat(i, j, num_vertices)
            M_mat += np.outer(u_vec, Tij).T

    return M_mat


def get_u_vec(i, j, n):
    u_vec = np.zeros(n*(n+1)//2)
    pos = (j-1) * n + i - j*(j-1)//2
    u_vec[pos-1] = 1
    return u_vec


def get_T_mat(i, j, n):
    Tij_mat = np.zeros((n, n))
    Tij_mat[i-1, j-1] = Tij_mat[j-1, i-1] = 1
    return np.ravel(Tij_mat)


def get_a_vec(i, n):
    a_vec = np.zeros(n*(n+1)//2)
    if i == 0:
        a_vec[np.arange(n)] = 1
    else:
        tmp_vec = np.arange(n-1, n-i-1, -1)
        tmp2_vec = np.append([i], tmp_vec)
        tmp3_vec = np.cumsum(tmp2_vec)
        a_vec[tmp3_vec] = 1
        end_pt = tmp3_vec[-1]
        a_vec[np.arange(end_pt, end_pt + n-i)] = 1

    return a_vec


def create_A_mat(n):
    A_mat = np.zeros((n+1, n*(n+1)//2))
    # A_mat[0, 0] = 1
    # A_mat[0, np.cumsum(np.arange(n, 0, -1))] = 1
    for i in range(0, A_mat.shape[0] - 1):
        A_mat[i, :] = get_a_vec(i, n)
    A_mat[n, 0] = 1
    A_mat[n, np.cumsum(np.arange(n, 1, -1))] = 1

    return A_mat


def create_b_mat(n):
    b_mat = np.zeros(n+1)
    b_mat[n] = n
    return b_mat


def create_G_mat(n):
    G_mat = np.zeros((n*(n-1)//2, n*(n+1)//2))
    tmp_vec = np.cumsum(np.arange(n, 1, -1))
    tmp2_vec = np.append([0], tmp_vec)
    tmp3_vec = np.delete(np.arange(n*(n+1)//2), tmp2_vec)
    for i in range(G_mat.shape[0]):
        G_mat[i, tmp3_vec[i]] = 1

    return G_mat





def single_gsp_exp(
    X,
    A,
    ratios,
    positions,
    metric_fn,
    select_metric='lowest',
    *,
    max_iter=1000,
    tau=None,
    labels=None,
    annot=True,
    save_dir_path=None,
    file_prefix='',
    save_ext='pdf',
    verbose: bool = False,
    disable_plotting: bool = False,
):
    if not isinstance(ratios, (list, tuple, np.ndarray)):
        ratios = [ratios]
    graph_save_path = None
    best = {}
    
    scores = np.zeros(len(ratios))
    for i, ratio in enumerate(ratios):
        ratio_beta = 1
        ratio_alpha = 1 / ratio
        try:
            ratio_L, _ = gl_sig_model(X, max_iter, ratio_alpha, ratio_beta)
        except AssertionError:
            if verbose: print(f"Inproperly learned L. Continuing...")
            continue
        if tau is not None: 
            ratio_L = thresholding(ratio_L, tau=tau)
        ratio_A_hat = to_adjacency(ratio_L)
        
        kwargs = filter_function_kwargs(
            metric_fn,
            A=A,
            A_hat=ratio_A_hat,
        )
        scores[i] = metric_fn(**kwargs)
        if select_metric == 'lowest':
            score_measure = scores[i] <= best.get('score', np.inf)
        elif select_metric == 'highest':
            score_measure = scores[i] >= best.get('score', -np.inf)
        else:
            raise ValueError("`select_metric` {select_metric} must be either 'lowest' or 'highest'")
        
        if score_measure:
            # print(ratio, scores[i], score_measure)
            best['ratio'] = ratio
            best['beta'] = ratio_beta
            best['alpha'] = ratio_alpha
            best['L'] = ratio_L
            best['A_hat'] = ratio_A_hat
            best['score'] = scores[i]

    if 'score' not in best:
        raise ValueError("All ratios threw an AssertionError!")
    ratio = best['ratio']
    beta = best['beta']
    alpha = best['alpha']
    L = best['L'] # A_hat before thresholding
    A_hat = best['A_hat'] 
    score = best['score']
    
    hd = hamming_distance(A, A_hat)
    info = {
        'metrics': {
            'score': score,
            'hd': hd,
        },
        'params': {
            'beta': beta,
            'alpha': alpha,
            'ratio': ratio,
            'tau': tau,
        }
    }
    
    title = f"GSP - Score: {np.round(score, 3)} HD: {hd}  Ratio: {np.round(ratio, 3)} Beta: {np.round(beta, 3)} Alpha: {np.round(alpha, 5)}"
    title += f" Tau: {np.round(tau,5)}" if tau is not None else ''

    if save_dir_path is not None:
        file_name = 'graph'
        file_name += f'_{file_prefix}-' if len(file_prefix) != 0  else ''
        file_name += f"_ratio={np.round(ratio, 2)}"
        file_name += f"_tau={tau}" if tau is not None else ''
        save_dir_path.mkdir(parents=True, exist_ok=True)
        graph_save_path = save_dir_path/f'{file_name}.{save_ext}'
        
    plot = plot_recovered(
        A=A_hat, 
        theta=L,
        positions=positions, 
        labels=labels,
        title=title, 
        sub_titles=['Recovered Graph', 'Recovered Laplacian Matrix'],
        save_path=graph_save_path,
        number_fmt='.2f',
        annot_kws = {"fontsize":9} ,
        vmin=None,
        vmax=None,
        annot=annot,
    )
    
    scores_df = pd.DataFrame(np.vstack([ratios, scores]).T, columns=['ratios', 'scores'])
    scores_fig, score_ax = plt.subplots(1,1)
    sns.lineplot(data=scores_df, x='ratios', y='scores', marker='o', ax=score_ax)
    if save_dir_path is not None:
        file_name = 'ratio-scores'
        file_name += f'_{file_prefix}-' if len(file_prefix) != 0  else ''
        file_name += f"_tau={tau}" if tau is not None else ''
        scores_fig.savefig(save_dir_path/f'{file_name}.{save_ext}')
    
    if not disable_plotting: 
        plt.show()

    return A_hat, L, info, plot, scores_df