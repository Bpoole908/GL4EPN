from pdb import set_trace

import numpy as np
import networkx as nx
from gglasso.helper.basic_linalg import Sdot
from gglasso.helper.model_selection import robust_logdet
from sklearn.utils.extmath import fast_logdet


def graph_precision(A, A_hat):
    """Computes the percentage of correct edges in A_hat that are in A
    
        Example:
            a = np.array([
                [0, 2, 2],
                [2, 0, 1],
                [2, 1, 0]
            ])

            b = np.array([
                [0, 3, 4],
                [3, 0, 2],
                [4, 2, 0]
            ])
            
            graph_precision(A=a, A_hat=b) -> precision of ~0.55
            
            graph_precision(A_hat=a, A=b) -> recall of 1
    
    """
    total = A_hat.sum()
    # Positive value means the edges of A were NOT in A_hat
    # Negative value means the edges of A_hat were NOT in A
    diff = A_hat - A
    not_in_A_hat = diff[diff > 0].sum()
    # not_in_A = abs(diff[diff < 0].sum())
    correct = total - not_in_A_hat
    # print(correct/2, total/2, correct/total)
    return correct / total if correct != 0 and total != 0 else 0


def graph_recall(A, A_hat):
    return graph_precision(A=A_hat, A_hat=A)


def graph_fscore(A, A_hat):
    """ Graph f-score
        Precision: Computes the percentage of correct edges in the predicted graph. Can be
            misleading if only 1 prediction is made and it is correct, yet ground truth
            has many more edges.
        Recall: Computes the percentaged of edges of the ground truth graph A that are in
            the predicted graph A_hat. This ensures all edges are in graph A_hat but says
            nothing about false edges.
    """
    prec = graph_precision(A=A, A_hat=A_hat)
    recall = graph_precision(A=A_hat, A_hat=A)
    return 2 * prec * recall / (prec + recall + np.finfo(float).eps) 


def graph_fbscore(A, A_hat, beta):
    """ Graph f-score
        Precision: Computes the percentage of correct edges in the predicted graph. Can be
            misleading if only 1 prediction is made and it is correct, yet ground truth
            has many more edges.
        Recall: Computes the percentaged of edges of the ground truth graph A that are in
            the predicted graph A_hat. This ensures all edges are in graph A_hat but says
            nothing about false edges.
    """
    prec = graph_precision(A=A, A_hat=A_hat)
    recall = graph_precision(A=A_hat, A_hat=A)
    return (1 + beta**2) * ( (prec * recall) / ( (beta**2*prec) + recall + np.finfo(float).eps ) )


def hamming_distance(A, A_hat, directed=False):
    """ Quick GED computation or hamming distance """
    # edge_diff = A1 - A2
    # total_diff = np.abs(edge_diff).sum()
    total_diff =  np.count_nonzero(A!=A_hat) 
    return total_diff if directed else total_diff / 2


def min_hd_metric(A, A_hat, min_edges, strict=False, directed=False):
    n_edges = np.sum(A_hat) if directed else np.sum(A_hat)/2
    A_locs = np.where(np.tril(A) >= 1)

    if strict and A_hat[A_locs].sum() < min_edges:
        return np.inf
    elif n_edges < min_edges:
        return np.inf
    return hamming_distance(A, A_hat)


def ged(A, A_hat):  
    G1 = nx.from_numpy_array(A)
    G2 = nx.from_numpy_array(A_hat)
    return nx.graph_edit_distance(G1, G2)


def ebic(R, theta, N, gamma, scaled=True, use_robust=True, verbose=False):
    (p,p) = R.shape
    
    # count upper diagonal non-zero entries
    E = (np.count_nonzero(theta) - p)/2
    
   
    ll1 = Sdot(R, theta)
    if use_robust:
        ll2 = robust_logdet(theta) 
    else: 
        l = fast_logdet(theta)
        ll2 = l[0]*l[1]
    nll = ll1 - ll2
    
    sample_pen = E * np.log(N) 
    node_pen = E*4*np.log(p)*gamma
    scale = N if scaled else 1
    score = scale*nll + sample_pen + node_pen
        
    return score


def unscaled_ebic_metric_fn(R, theta, N, gamma=0.5):
    return ebic(S=R, theta=theta, N=N, gamma=gamma)


def aic(S, theta, N, use_robust=True, verbose=False):
    (p,p) = S.shape
        
    # count upper diagonal non-zero entries
    E = (np.count_nonzero(theta) - p)/2
    ll1 = Sdot(S, theta)
    if use_robust:
        ll2 = robust_logdet(theta) 
    else: 
        l = fast_logdet(theta)
        ll2 = l[0]*l[1]
    ll = ll1 - ll2

    aic = ll + E
    if verbose:
        print(f"AIC: {aic}")
        print(f"LogLik: {ll:.2f} = {ll1:.2f} - {ll2:.2f}")
        print(f"LogLik_N: {ll*N:.2f} = {N} * {ll:.2f}")
        print(f"E: {E}")
    
    return aic


def graph_log_likelihood(S, theta):
    return Sdot(S, theta) - robust_logdet(theta)