import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _softmax(X: np.ndarray, axis: int = 0) -> np.ndarray:
    X = X - X.max(axis=axis, keepdims=True)
    eX = np.exp(X)
    return eX / eX.sum(axis=axis, keepdims=True)


def compute_p(f: np.ndarray,
              b: np.ndarray,
              t: np.ndarray) -> np.ndarray:

    logits = f[:, None] * t[None, :] + b[:, None]
    return _softmax(logits, axis=0)


# ── Fitness update, joint across locations ──────────────────────────────────

def _update_fitness_soft_joint(locations, H, f_init, b_init, t, lambda_f, bfgs_options=None):
    """
    Joint generalization of _update_fitness_soft: fits one shared f across locations and
    one b_z per location, so that softmax(f*t_z + b_z) ~= H_z for every
    location simultaneously. Same pack/unpack/objective/gradient pattern
    as the single-location version, replicated per location and summed
    for f's gradient (shared -- appears in every location's term) but
    kept separate for each b_z's gradient (local -- appears in only one
    location's term).
    """
    r = f_init.shape[0]
    if r <= 1:
        # Only the reference component exists -- there is nothing free to
        # fit (f_1=0, b_1^z=0 already fixes everything). Matches the
        # equivalent guard in joint_nmf.py's _fitness_step; without it,
        # BFGS is called on a zero-length parameter vector and scipy
        # raises inside its convergence-norm check.
        return f_init.copy(), {loc: b_init[loc].copy() for loc in locations}

    def pack(f, b):
        return np.concatenate([f[1:]] + [b[loc][1:] for loc in locations])

    def unpack(x):
        f = np.concatenate([[0.0], x[:r - 1]])
        b = {}
        offset = r - 1
        for loc in locations:
            b[loc] = np.concatenate([[0.0], x[offset:offset + r - 1]])
            offset += r - 1
        return f, b

    def obj_and_grad(x):
        f, b = unpack(x)
        obj = 0.0
        grad_f = np.zeros(r)
        grad_b = {loc: np.zeros(r) for loc in locations}

        for loc in locations:
            P = compute_p(f, b[loc], t[loc])          # (r, s_loc)
            diff = P - H[loc]
            obj += (lambda_f / 2.0) * np.sum(diff ** 2)

            dL_dP = lambda_f * diff
            weighted = dL_dP * P
            col_sum = weighted.sum(axis=0, keepdims=True)
            delta = weighted - P * col_sum

            grad_f += (delta * t[loc][None, :]).sum(axis=1)   # shared -> accumulate over z
            grad_b[loc] = delta.sum(axis=1)                   # local -> this location only

        grad = pack(grad_f, grad_b)  # reusing pack() purely to slice off index 0 consistently
        return obj, grad

    x0 = pack(f_init, b_init)
    result = minimize(
        obj_and_grad, x0, method='BFGS', jac=True,
        options=bfgs_options or {'maxiter': 1000, 'gtol': 1e-8},
    )
    return unpack(result.x)

