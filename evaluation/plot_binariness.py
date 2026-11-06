import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_binariness(W, title='Binariness of W'):
    """
    Plot distribution of W entries to assess how binary they are.
    Binary W should have entries close to 0 or 1.
    """
    values = W.flatten()

    fig, axes = plt.subplots(1, 2, figsize=(5, 2.0))

    # ── Fraction of entries close to 0 or 1 ──────────────────────────────────
    near_0   = np.mean(values < 0.4)
    near_1   = np.mean(values > 0.6)
    in_between = 1 - near_0 - near_1

    axes[0].bar(['< 0.4\n', '0.4 - 0.6\n(ambiguous)', '> 0.6\n'],
                [near_0, in_between, near_1],
                color=['steelblue', 'orange', 'steelblue'])
    axes[0].set_ylabel('Fraction of entries')
    axes[0].set_title(f'Binariness\n({near_0+near_1:.1%} binary)')
    for i, v in enumerate([near_0, in_between, near_1]):
        axes[0].text(i, v + 0.01, f'{v:.1%}', ha='center', fontsize=9)

    # ── Per-component binariness ──────────────────────────────────────────────
    if hasattr(W, 'columns'):
        comp_names = W.columns
        W_vals     = W.values
    else:
        comp_names = [f'C{i}' for i in range(W.shape[1])]
        W_vals     = W

    binariness = [np.mean((W_vals[:, k] < 0.4) | (W_vals[:, k] > 0.6))
                  for k in range(W_vals.shape[1])]

    axes[1].bar(comp_names, binariness, color='steelblue')
    axes[1].axhline(0.9, color='red', linestyle='--', label='90% threshold')
    axes[1].set_xlabel('Component')
    axes[1].set_ylabel('Fraction binary entries')
    axes[1].set_title('Binariness per component')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_ylim(0, 1.05)
    #put bottom right
    axes[1].legend(loc='lower right')

    #plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()




# ── Usage ─────────────────────────────────────────────────────────────────────
#plot_binariness(result.W, title='Binariness of W')
