import pandas as pd
import re
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
# ── Date parsing helper ──────────────────────────────────────────────────────
# Duplicated (not imported) from soft_dynamics_nmf's inline top-of-function
# block, since it isn't factored out as a standalone helper there -- kept
# byte-for-byte identical in logic so date parsing behaves the same way for
# each location as it would for a single-location call.

def _parse_dates(A_loc):
    A_loc = pd.DataFrame(A_loc)
    A_loc.columns = [col.strftime('%Y_%m_%d') if hasattr(col, 'strftime') else str(col)
                      for col in A_loc.columns]
    A_loc.columns = [re.search(r'(\d{4}_\d{2}_\d{2})', col).group(1) for col in A_loc.columns]
    A_loc.columns = pd.to_datetime(A_loc.columns, format='%Y_%m_%d')
    A_loc = A_loc.sort_index(axis=1)
    return A_loc, pd.to_datetime(A_loc.columns)


# ──── Prepare output: binarized signatures ───
def binarize_signature_matrix(W: np.ndarray) -> np.ndarray:
    """Binarize signature matrix W"""
    W_final = W.astype(float).copy()

    W_final[W_final >= 0.5] = 1.0
    W_final[W_final < 0.5] = 0.0
    return W_final


def denovo_target_jaccard_hungarian(result, W_aligned, known_variants, target_variant):
    """Jaccard between the target's true signature and its Hungarian-
    assigned predicted column, for a fully unconstrained ("de novo") fit."""
    gt_variants = list(known_variants) + [target_variant]
    target_idx = len(known_variants)

    W_pred = binarize_signature_matrix(result.W)
    W_gt = binarize_signature_matrix(W_aligned[gt_variants].values)

    dist = cdist(W_pred.T, W_gt.T, metric='jaccard')
    sim = 1 - dist

    row_ind, col_ind = linear_sum_assignment(-sim)
    assigned = dict(zip(col_ind, sim[row_ind, col_ind]))
    return float(assigned.get(target_idx, np.nan))