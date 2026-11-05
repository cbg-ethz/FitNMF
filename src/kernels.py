import numpy as np
import pandas as pd

# ── Similarity matrix ──────────────────────────────────────────────────────────

def _make_similarity_matrix(time_points: np.ndarray,
                             bandwidth: float
                             ) -> np.ndarray:
    """
    Gaussian kernel similarity matrix between time points.

    S_ij = exp(-(t_i - t_j)^2 / bandwidth^2)

    normalized by the mean row sum

    Parameters
    ----------
    time_points : np.ndarray, shape (n_samples,)
        Numeric time points (days).
    bandwidth : float
        Kernel bandwidth in same units as time_points.

    Returns
    -------
    S : np.ndarray, shape (n_samples, n_samples)
        normalized by the mean row sum similarity matrix.
    """
    diff = time_points.reshape(-1, 1) - time_points.reshape(1, -1) # (n, n)
    S = np.exp(-(diff ** 2) / (bandwidth ** 2 + 1e-8))  # Gaussian kernel
    S = S /S.sum(axis=1).mean() # symmetric normalization 
    return S


def _make_laplacian(S: np.ndarray):
    """
    Graph Laplacian L = D - S where D_ii = sum_j S_ij.

    After row-normalization of S, D = I, so L = I - S.

    Parameters
    ----------
    S : np.ndarray, shape (n, n)
        normalized by the mean row sum similarity matrix.

    Returns
    -------
    D : np.ndarray, shape (n, n)
        Diagonal degree matrix.
    L : np.ndarray, shape (n, n)
        Graph Laplacian D - S.
    """
    D = np.diag(S.sum(axis=1))
    L = D - S
    return D, L
