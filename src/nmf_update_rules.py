import pandas as pd
import re
import numpy as np


def _update_W_joint(locations, data, M, W, H, lambda_w, C,
                     eps=1e-8):
    m, rank = W.shape
    recon_num = np.zeros((m, rank))
    recon_den = np.zeros((m, rank))
    for loc in locations:
        WH = W @ H[loc]
        recon_num += (data[loc] * M[loc]) @ H[loc].T
        recon_den += (WH * M[loc]) @ H[loc].T

    bin_num = 3.0 * lambda_w * C * (W ** 2)
    bin_den = lambda_w * (2.0 * W ** 3 + (C ** 2) * W)

    W1 = recon_num + bin_num
    W2 = recon_den + bin_den
    return W * (W1 / (W2 + eps))


def _update_W_masked_option1_joint(locations, data, M, W, H, Q, Q_mask,
                                    lambda_w, lambda_mask, C, 
                                     eps=1e-8):
    m, rank = W.shape
    recon_num = np.zeros((m, rank))
    recon_den = np.zeros((m, rank))
    for loc in locations:
        WH = W @ H[loc]
        recon_num += (data[loc] * M[loc]) @ H[loc].T
        recon_den += (WH * M[loc]) @ H[loc].T


    bin_num = 3.0 * lambda_w * C * (W ** 2)
    bin_den = lambda_w * (2.0 * W ** 3 + (C ** 2) * W)

    mask_num = lambda_mask * Q * (Q_mask ** 2)
    mask_den = lambda_mask * W * (Q_mask ** 2)


    W1 = recon_num + bin_num + mask_num
    W2 = recon_den + bin_den + mask_den 
    return W * (W1 / (W2 + eps))


def _update_W_masked_option2_joint(locations, data, M, W, H, Q, Q_mask,
                                    lambda_w, C, eps=1e-8):
    m, rank = W.shape
    recon_pos = np.zeros((m, rank))
    recon_neg = np.zeros((m, rank))
    for loc in locations:
        WH = W @ H[loc]
        recon_pos += (data[loc] * M[loc]) @ H[loc].T
        recon_neg += (WH * M[loc]) @ H[loc].T


    bin_pos = 3.0 * lambda_w * C * (W ** 2)
    bin_neg = lambda_w * (2.0 * W ** 3 + (C ** 2) * W)

    W_free = W * ((recon_pos + bin_pos) / (recon_neg + bin_neg  + eps))
    return Q_mask * Q + (1.0 - Q_mask) * W_free




def _update_H_soft(A, M, W, H, P, S, D,
                   lambda_h, lambda_f, eps=1e-8):
    """
    MU update for H with:
        - graph Laplacian smoothing  (lambda_h)
        - soft dynamics penalty      (lambda_f)

    Gradient:
        dL/dH = -W^T(M⊙(A-WH)) + lambda_h*(HD-HS) + lambda_f*(H-P)

    MU split:
        num = W^T(M⊙A)  + lambda_h*(H@S) + lambda_f*P
        den = W^T(M⊙WH) + lambda_h*(H@D) + lambda_f*H
    """
    WH  = W @ H
    num = W.T @ (A * M)  + lambda_h * (H @ S) + lambda_f * P
    den = W.T @ (WH * M) + lambda_h * (H @ D) + lambda_f * H

    return H * (num / (den + eps))

