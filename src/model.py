"""
joint_soft_selection_nmf.py

Joint NMF: one shared signature matrix W and one shared
fitness vector f across all locations, one abundance matrix H_z and one
offset vector b_z per location.


Each location's time axis t_z is measured in days since ONE SHARED
  reference date -- the earliest date across ALL locations -- not since
  that location's own first sample. This matters because f is shared: for "the same
  fitness value f_k applies in every location" to mean anything
  scientifically, t=0 has to refer to the same calendar date everywhere.
  If each location's clock reset independently, BFGS would be fitting
  one f against several (potentially incompatible) time axes.

"""
#import re
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Import relevant functions
from .nmf_update_fitness import compute_p,  _update_fitness_soft_joint
from .kernels import _make_similarity_matrix, _make_laplacian
from .nmf_update_rules import _update_W_masked_option2_joint, _update_W_joint, _update_W_masked_option1_joint, _update_H_soft
from .helpers import _parse_dates



@dataclass
class JointSoftDynamicsNMFResult:
    W:                np.ndarray            # (m, r) shared signature matrix binarized
    W_unnormalized: np.ndarray            # (m, r) unnormalized signature matrix
    f:                Optional[np.ndarray]            # shared fitness vector, PER DAY (f[0] == 0)
    weekly_growth_adv: Optional[np.ndarray]           # f * 7 -- per-week units, comparable to covvfit's Estimate
    b:                Optional[dict]                  # location -> (r,) offsets, b[loc][0] == 0
    H:                dict                  # location -> (r, s_loc) abundances
    H_unnormalized:  dict                  # location -> (r, s_loc) unnormalized abundances
    P:                Optional[dict]                  # location -> (r, s_loc) softmax(f*t_loc+b_loc)
    t:                dict                  # location -> (s_loc,) time axis, days since the shared origin
    n_iter:           int
    objective_values: np.ndarray
    converged:        bool
    residuals:        dict                  # location -> (m, s_loc) masked residual
    all_runs:         list
    best_run_index:   int


# ── Single restart ───────────────────────────────────────────────────────────

def _single_run_joint(data, M, S, D, L, t, locations, rank,
                       lambda_w, lambda_h, lambda_f,
                       masking_option, Q, Q_mask, lambda_mask,
                       max_iter, rel_tol, n_mu_steps, bfgs_options,
                       rng, run_idx, verbose, C, eps=1e-8):

    m = data[locations[0]].shape[0]

    W = rng.uniform(0, 1, size=(m, rank))

    H = {loc: rng.uniform(0, 1, size=(rank, data[loc].shape[1])) for loc in locations}

    if lambda_f > 0:
        f = np.concatenate([[0.0], rng.uniform(-1.0,  1.0, size=rank-1)])
        b = {loc: np.concatenate([[0.0], rng.uniform(-10.0, 10.0, size=rank - 1)]) for loc in locations}

    else:
        f = np.zeros(rank)
        b = {loc: np.zeros(rank) for loc in locations}


    # Warm-up: H update per location 
    for loc in locations:

        H[loc] = np.maximum(
            _update_H_soft(data[loc], M[loc], W, H[loc], np.zeros((rank, data[loc].shape[1])),
                            S[loc], D[loc], lambda_h, lambda_f=0, eps=1e-8),
            eps)

    if masking_option == 2:
        W = np.maximum(_update_W_masked_option2_joint(locations, data, M, W, H, Q, Q_mask, lambda_w, C, eps), eps)
    else:
        W = np.maximum(_update_W_joint(locations, data, M, W, H, lambda_w, C, eps), eps)

    Q_arg = Q if masking_option == 1 else None
    Qm_arg = Q_mask if masking_option == 1 else None
    lm_arg = lambda_mask if masking_option == 1 else 0.0

    objective_values = []
    p_obj = np.inf
    c_obj = np.inf
    n_iter = 0
    converged = False


    P = {loc: compute_p(f, b[loc], t[loc]) for loc in locations}


    while n_iter < max_iter:
        #P = {loc: compute_p(f, b[loc], t[loc]) for loc in locations}

        # Step 1: H update, per location.
        for _ in range(n_mu_steps):
            for loc in locations:
                H[loc] = _update_H_soft(data[loc], M[loc], W, H[loc], P[loc], S[loc], D[loc], lambda_h, lambda_f, eps)
                H[loc] = np.maximum(H[loc], eps)
    

        # Step 2: W update.
        for _ in range(n_mu_steps):
            if masking_option == 0:
                W = _update_W_joint(locations, data, M, W, H, lambda_w, C, eps)
            elif masking_option == 1:
                W = _update_W_masked_option1_joint(locations, data, M, W, H, Q, Q_mask, lambda_w, lambda_mask, C, eps)
            elif masking_option == 2:
                W = _update_W_masked_option2_joint(locations, data, M, W, H, Q, Q_mask, lambda_w, C, eps)
            W = np.maximum(W, eps)


        # Step 3: f, {b_z} update via joint BFGS.
        if lambda_f > 0:
            f, b = _update_fitness_soft_joint(locations, H, f, b, t, lambda_f, bfgs_options)

        # Objective.
        P = {loc: compute_p(f, b[loc], t[loc]) for loc in locations}
        rec_total = 0.0
        smooth_total = 0.0
        soft_total = 0.0
        for loc in locations:
            R = (data[loc] - W @ H[loc]) * M[loc]
            rec_total += np.sum(R ** 2)
            smooth_total += lambda_h * np.trace(H[loc] @ L[loc] @ H[loc].T)
            soft_total += lambda_f * np.sum((H[loc] - P[loc]) ** 2)
        bin_total = lambda_w * np.sum((W * (W - C)) ** 2)

        mask_term = 0.0
        if Q_arg is not None and Qm_arg is not None and lm_arg > 0:

            mask_term = lm_arg * 0.5 * np.sum(((W - Q_arg) * Qm_arg) ** 2)

        c_obj = 0.5 * rec_total + 0.5 * bin_total + 0.5 * smooth_total + 0.5 * soft_total + mask_term
        objective_values.append(c_obj)

        rel_change = (np.inf if (np.isinf(p_obj) or not np.isfinite(c_obj))
                      else abs(p_obj - c_obj) / (0.5 * (abs(p_obj) + abs(c_obj)) + eps))

        if verbose and n_iter % 50 == 0:
            print(f"run {run_idx + 1}  iter {n_iter:4d}  obj={c_obj:.6f}  f={np.round(f, 3)}")

        p_obj = c_obj
        n_iter += 1

        if rel_change < rel_tol:
            converged = True
            break

    P_final = {loc: compute_p(f, b[loc], t[loc]) for loc in locations}
    residuals = {loc: (data[loc] - W @ H[loc]) * M[loc] for loc in locations}

    return dict(
        W=W.copy(), f=f.copy(), b={loc: v.copy() for loc, v in b.items()},
        H={loc: v.copy() for loc, v in H.items()}, P=P_final,
        n_iter=n_iter, objective_values=np.array(objective_values),
        converged=converged, final_obj=c_obj, residuals=residuals,
    )





# ── Top-level entry point ────────────────────────────────────────────────────

def joint_soft_dynamics_nmf(
        data:           dict,      # location -> matrix, same format soft_dynamics_nmf's A takes
        rank:           int,
        masking_option: int = 0,
        Q:              Optional[np.ndarray] = None,       # shared across locations (pins W, which is shared)
        Q_mask:         Optional[np.ndarray] = None,
        max_iter:       int = 5000,
        rel_tol:        float = 1e-4,
        n_run:          int = 10,
        lambda_w:       float = 1.0,
        lambda_h:       float = 1.0,
        lambda_f:       float = 1.0,
        lambda_mask:    float = 0.0,
        bandwidth:      float = 10.0,
        n_mu_steps:     int = 1,
        bfgs_options:   Optional[dict] = None,
        n_jobs:         int = 1,
        random_state:   Optional[int] = None,
        verbose:        bool = False,
        eps:            float = 1e-8,
        n_half_frac:    float = 0.10,
        C:              Optional[np.ndarray] = None
) -> JointSoftDynamicsNMFResult:
    """
    Joint (multi-location) soft-dynamics NMF: shared W and shared fitness
    vector f, one H_z and one offset b_z per location.

    Parameters
    ----------
    data
    rank, max_iter, rel_tol, n_run, lambda_w, lambda_h, bandwidth,
    n_mu_steps, n_jobs, random_state, verbose, eps,
    masking_option, Q, Q_mask, lambda_mask :
    lambda_f : float
    bfgs_options : dict or None


    Returns
    -------
    JointSoftDynamicsNMFResult
    """
    locations = list(data.keys())
    if len(locations) == 0:
        raise ValueError("data must have at least one location")

    parsed = {loc: _parse_dates(data[loc]) for loc in locations}
    A_sorted = {loc: parsed[loc][0] for loc in locations}
    cols = {loc: parsed[loc][1] for loc in locations}

    # Shared time origin across ALL locations -- see module docstring.
    t0 = min(c[0] for c in cols.values())
    t = {loc: (cols[loc] - t0).days.to_numpy(dtype=float) for loc in locations}

    A_arr = {loc: np.array(A_sorted[loc], dtype=float) for loc in locations}
    m = A_arr[locations[0]].shape[0]
    for loc in locations:
        if A_arr[loc].shape[0] != m:
            raise ValueError(
                f"location {loc!r} has {A_arr[loc].shape[0]} feature rows, expected {m} "
                f"(every location must share the same feature/mutation rows for a shared W)")

    min_samples = min(A_arr[loc].shape[1] for loc in locations)
    if rank > min(m, min_samples):
        raise ValueError(f"rank={rank} exceeds min(n_features, min_samples_per_location)={min(m, min_samples)}")
    if masking_option not in {0, 1, 2}:
        raise ValueError("masking_option must be 0, 1, or 2")
    if masking_option > 0 and (Q is None or Q_mask is None):
        raise ValueError("Q and Q_mask required when masking_option > 0")
    if masking_option > 0:
        Q = np.array(Q, dtype=float)
        Q_mask = np.array(Q_mask, dtype=float)

    M = {loc: (~np.isnan(A_arr[loc])).astype(float) for loc in locations}
    A_filled = {loc: np.where(np.isnan(A_arr[loc]), 0.0, A_arr[loc]) for loc in locations}

    S, D, L = {}, {}, {}
    for loc in locations:
        S[loc] = _make_similarity_matrix(t[loc], bandwidth)
        D[loc], L[loc] = _make_laplacian(S[loc])

    seed_seq = np.random.SeedSequence(random_state)
    seeds = seed_seq.spawn(n_run)
    rngs = [np.random.default_rng(seed) for seed in seeds]

    if C is None:
        C = _evidence_target(locations, M, n_half_frac)
    else:
        C = np.asarray(C, dtype=float).reshape(-1, 1)   # (m, 1); n_half_frac is then ignored


    lambda_w_eff    = lambda_w
    lambda_mask_eff = lambda_mask

    all_runs = Parallel(n_jobs=n_jobs)(
        delayed(_single_run_joint)(
            A_filled, M, S, D, L, t, locations, rank,
            lambda_w_eff, lambda_h, lambda_f,
            masking_option, Q, Q_mask, lambda_mask_eff,
            max_iter, rel_tol, n_mu_steps, bfgs_options,
            rngs[i], i, verbose, C=C, eps=eps)
        for i in range(n_run)
    )

    best_idx = int(np.argmin([r['final_obj'] for r in all_runs]))
    best = all_runs[best_idx]
    n_converged = sum(r['converged'] for r in all_runs)
    if n_converged == 0:
        warnings.warn(f"No run converged within max_iter={max_iter}.", RuntimeWarning)

    
    H_out = {loc: h / np.sum(h, axis=0, keepdims=True) for loc, h in best['H'].items()}




    return JointSoftDynamicsNMFResult(
        W= (best['W'] >= 0.5).astype(int), 
        W_unnormalized=best['W'],
        f=best['f'] if lambda_f >0 else None, 
        weekly_growth_adv=best['f'] * 7.0 if lambda_f >0 else None,   # t is raw days, so f is already per-day -- just *7, no /t_max, 
        b=best['b'] if lambda_f >0 else None, 
        H=H_out, 
        H_unnormalized=best['H'],
        P=best['P'] if lambda_f >0 else None, 
        t=t,
        n_iter=best['n_iter'], 
        objective_values=best['objective_values'],
        converged=best['converged'], 
        residuals=best['residuals'],
        all_runs=all_runs, best_run_index=best_idx,
    )



def fit_model_denovo_joint(data_window, rank, **nmf_kwargs):
    """Fully unconstrained joint NMF -- no known-variant pinning at all."""
    result = joint_soft_dynamics_nmf(
        data_window, rank=rank, masking_option=0, **nmf_kwargs)
    return result



def _evidence_target(locations, M, n_half_frac, eps=1e-8):
    """a_i: how far the binarization penalty may push row i toward 'present'.
    n_i counts adequately-covered observations in the current window; n_half is
    scaled to the fit's own median so a is comparable across combo sizes and
    window lengths (an absolute n_half would be inert for six-plant fits and
    active for single-plant ones)."""
    n_obs = sum(M[loc].sum(axis=1) for loc in locations)
    pos   = n_obs[n_obs > 0]
    n_ref = np.median(pos) if pos.size else 1.0
    return (n_obs / (n_obs + max(n_half_frac * n_ref, eps)))[:, None]
