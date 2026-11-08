import numpy as np
import pandas as pd
from itertools import product
from joblib import Parallel, delayed, parallel_backend

from src.model import joint_soft_dynamics_nmf, _evidence_target


def _build_pooled_coords(data_arr):
    """
    Pool every location's non-missing (row, col) coordinates into one flat
    list tagged by location, so a single K-fold split can be drawn across
    ALL locations at once.
    """
    pooled = []
    for loc, arr in data_arr.items():
        for r, c in np.argwhere(~np.isnan(arr)):
            pooled.append((loc, r, c))
    return pooled


def _evaluate_single_joint(k, fold_coords, rank, data_arr,
                            lambda_w, lambda_h, lambda_f, n_run,
                            bandwidth, cols, idx,
                            max_iter, rel_tol, masking_option=0,
                            Q=None, Q_mask=None, lambda_mask=0.0, random_state=None):
    """Evaluate one (fold, lambda_w, lambda_h[, lambda_f]) combination for
    the joint multi-location model: hold out this fold's (location, row,
    col) triples as NaN before fitting, fit on what's left, score
    reconstruction error at the held-out triples against the
    ORIGINAL (pre-masking) values."""

    data_fold = {loc: arr.copy() for loc, arr in data_arr.items()}
    test_by_loc = {}
    for loc, r, c in fold_coords:
        data_fold[loc][r, c] = np.nan
        test_by_loc.setdefault(loc, []).append((r, c))

    data_fold_df = {
        loc: pd.DataFrame(arr_fold, index=idx[loc], columns=cols[loc])
        for loc, arr_fold in data_fold.items()
    }
    M_full = {loc: (~np.isnan(data_arr[loc])).astype(float) for loc in data_arr}

    C_full = _evidence_target(data_arr.keys(), M_full, 0.1) # binarization target


    kwargs = dict(data=data_fold_df, rank=rank,
                lambda_w=lambda_w, lambda_h=lambda_h, lambda_f=lambda_f,
                n_run=n_run, bandwidth=bandwidth, n_jobs=1,
                max_iter=max_iter, rel_tol=rel_tol,
                masking_option=masking_option, Q=Q, Q_mask=Q_mask,
                lambda_mask=lambda_mask, C=C_full, random_state=(random_state or 0) + k)


    result = joint_soft_dynamics_nmf(**kwargs)

    total_sq, total_n = 0.0, 0.0
    per_loc_mse = {}
    for loc, pairs in test_by_loc.items():
        rows = np.array([p[0] for p in pairs])
        colsi = np.array([p[1] for p in pairs])
        reconstructed_loc = result.W_unnormalized @ result.H_unnormalized[loc]
        true_vals = data_arr[loc][rows, colsi]      # ORIGINAL, unmasked values
        pred_vals = reconstructed_loc[rows, colsi]
        sq = np.sum((true_vals - pred_vals) ** 2)
        total_sq += sq
        total_n += len(pairs)
        per_loc_mse[loc] = float(sq / len(pairs))

    mse = float(total_sq / total_n) if total_n > 0 else np.nan

    print(f"  Fold {k+1}: lambda_w={lambda_w}, lambda_h={lambda_h}, "
        f"lambda_f={lambda_f}  mse={mse:.4f}")  

    return {
        'fold':                      k + 1,
        'n_variants':                rank,
        'lambda_w':                  lambda_w,
        'lambda_h':                  lambda_h,
        'lambda_f':                  lambda_f,
        'squared_error_loss':        mse,
        'squared_error_loss_by_loc': per_loc_mse,
    }


def joint_nmf_cross_validation(data, rank, lambda_w_values, lambda_h_values, lambda_f_values,
                                K, n_run, bandwidth,
                                n_jobs=1, output_csv=None,
                                max_iter=5000, rel_tol=1e-3,
                                masking_option=0, Q=None, Q_mask=None,
                                lambda_mask=0.0, random_state=None):
    """
    K-fold cross-validation for the joint multi-location model, with full
    parallelism over folds x lambda combinations.

    Parameters
    ----------
    data            : dict[str, DataFrame]   location -> mutation frequency matrix
    rank            : int                     number of variants
    lambda_w_values : list                    candidate lambda_w values
    lambda_h_values : list                    candidate lambda_h values
    lambda_f_values : list or None            candidate lambda_f values
    K               : int                     number of CV folds
    n_run           : int                     NMF random restarts per fit
    bandwidth       : float                   Gaussian kernel bandwidth

    n_jobs          : int                     number of parallel jobs (-1 = all cores)
    output_csv      : str or None             if provided, saves results to CSV
    masking_option, Q, Q_mask, lambda_mask :  passed straight through to
                                               joint_soft_dynamics_nmf, for
                                               cross-validating a masked/pinned fit
    random_state    : int or None             seeds the fold shuffle (reproducibility --
                                               the original script didn't expose this)

    Returns
    -------
    results_df  : pd.DataFrame  per-fold results
    cv_summary  : pd.DataFrame  mean error per lambda combination, sorted
    """
    locations = list(data.keys())


    cols, idx, data_arr = {}, {}, {}
    for loc in locations:
        df = data[loc]
        parsed_cols = pd.to_datetime(pd.Index(df.columns.astype(str)).str.replace("_", "-"), format="%Y-%m-%d")
        order = np.argsort(parsed_cols.values)
        df_sorted = df.iloc[:, order]
        cols[loc] = parsed_cols[order]
        idx[loc] = df_sorted.index
        data_arr[loc] = np.array(df_sorted, dtype=float)

    pooled_coords = _build_pooled_coords(data_arr)
    n_obs = len(pooled_coords)

    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(n_obs)
    folds = np.array_split(shuffled, K)
    fold_coord_lists = [[pooled_coords[i] for i in fold] for fold in folds]

    lambda_combinations = list(product(lambda_w_values, lambda_h_values, lambda_f_values))

    print(f"Running {K}-fold CV over "
          f"{len(lambda_combinations)} lambda combinations "
          f"using {n_jobs} jobs...")


    all_jobs = [
            (k, fold_coord_lists[k], lambda_w, lambda_h, lambda_f)
            for k in range(K)
            for lambda_w, lambda_h, lambda_f in lambda_combinations
        ]


    n_total = len(all_jobs)
    print(f"Running {K}-fold CV: "
          f"{len(lambda_combinations)} lambda combinations x {K} folds "
          f"= {n_total} total jobs using {n_jobs} workers...")

    with parallel_backend('loky', inner_max_num_threads=1):
        results_list = Parallel(n_jobs=n_jobs)(
            delayed(_evaluate_single_joint)(
                k, fold_coords, rank, data_arr,
                lambda_w, lambda_h, lambda_f, n_run,
                bandwidth, cols, idx, max_iter, rel_tol,
                masking_option, Q, Q_mask, lambda_mask, random_state)
            for k, fold_coords, lambda_w, lambda_h, lambda_f in all_jobs
        )
    results_df = pd.DataFrame(results_list)

    if output_csv is not None:
        results_df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

    group_cols = ['lambda_w', 'lambda_h', 'lambda_f']
    cv_summary = (results_df
                  .groupby(group_cols)['squared_error_loss']
                  .agg(['mean', 'std'])
                  .reset_index()
                  .rename(columns={'mean': 'mean_error', 'std': 'std_error'})
                  .sort_values('mean_error')
                  .reset_index(drop=True))

    return results_df, cv_summary


def best_lambda(cv_summary):
    """Return lambda combination with lowest mean CV error."""
    best = cv_summary.iloc[0]
    if 'lambda_f' in cv_summary.columns:
        return best['lambda_w'], best['lambda_h'], best['lambda_f']
    return best['lambda_w'], best['lambda_h']

