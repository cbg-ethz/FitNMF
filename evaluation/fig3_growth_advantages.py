"""
fig3_growth_advantages.py

Figure 3 of the manuscript: growth advantages inferred by FitNMF (Zurich WWTP, 2021-2024).
  (A) weekly log growth advantages 7(f_i - f_j) of co-circulating variant pairs, FitNMF vs. Covvfit,
      with Spearman rho and a two-sided label-permutation p-value;
  (B) FitNMF weekly growth advantage (%) of the variant in each row relative to the variant in each
      column, for co-circulating pairs only.

FitNMF components are matched to CoV-Spectrum signatures by Jaccard similarity of the binarized
signatures; if two components match the same variant, the one with the higher similarity is used.

Usage (notebook):
    from evaluation.fig3_growth_advantages import make_figure3
    stats = make_figure3(result_zh, data_zh, W_covspectrum,
                         covvfit_path=ROOT / 'results' / 'Covvfit_comparison' / 'results_covvfit',
                         out_dir=ROOT / 'results' / 'real_data')
"""
from itertools import combinations, islice, permutations
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
from matplotlib.transforms import Bbox
from scipy.spatial.distance import cdist
from scipy.stats import rankdata, spearmanr

plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                     'pdf.fonttype': 42, 'ps.fonttype': 42})

# Order of variants. A pair is always "later variant / earlier variant" in this order, which fixes the
# sign of each pairwise difference in both panels. The Spearman rho depends on this orientation:
# this order gives rho = 0.82, p = 1.4e-4 (as in the manuscript); with BA.2.75 before XBB.1.5 it is
# rho = 0.81, p = 2.8e-6.
VARIANTS_SC2 = ['B.1.1.7', 'B.1.617.2', 'BA.1', 'BA.2', 'BA.5',
                'BQ.1.1', 'XBB.1.5', 'BA.2.75', 'EG.5', 'JN.1']

# Groups of variants that circulated at the same time; only pairs within a group are compared.
COCIRCULATING_SC2 = [
    ('B.1.1.7', 'B.1.617.2'),
    ('B.1.617.2', 'BA.1', 'BA.2'),
    ('BA.2', 'BA.4', 'BA.5'),
    ('BA.4', 'BA.5', 'BA.2.75', 'BQ.1.1'),
    ('BA.5', 'BA.2.75', 'BQ.1.1', 'XBB.1.5', 'XBB.1.9'),
    ('BA.2.75', 'BQ.1.1', 'XBB.1.5', 'XBB.1.9', 'XBB.1.16', 'XBB.2.3'),
    ('BQ.1.1', 'XBB.1.5', 'XBB.1.9', 'XBB.1.16', 'XBB.2.3', 'EG.5'),
    ('XBB.1.5', 'XBB.1.9', 'XBB.1.16', 'XBB.2.3', 'EG.5', 'BA.2.86', 'JN.1'),
]

FS_LABEL, FS_TICK, FS_PAIR, FS_STAT = 16, 14, 11, 13          # panel A
FS_HM_LABEL, FS_HM_TICK, FS_HM_ANNOT, FS_HM_CBAR = 16, 14, 13, 12   # panel B
FS_PANEL = 22


# ── Matching FitNMF components to reference variants ───────────────────────

def match_fitness_to_variants(result, mutation_index, W_covspectrum,
                              binarize_threshold=0.5, unassigned_threshold=0.5):
    """{variant: 7*f} for every component whose best Jaccard similarity is >= unassigned_threshold.
    Returns (fitness_by_variant, sim_df)."""
    W = np.asarray(result.W)
    if len(mutation_index) != W.shape[0]:
        raise ValueError(f"mutation_index has {len(mutation_index)} entries, result.W has {W.shape[0]} rows")
    comps = [f'Component_{k}' for k in range(W.shape[1])]
    W_bin = pd.DataFrame((W >= binarize_threshold).astype(int), index=mutation_index, columns=comps)

    muts = sorted(set(W_bin.index) | set(W_covspectrum.index))
    sim = 1 - cdist(W_bin.reindex(muts, fill_value=0).values.T,
                    W_covspectrum.reindex(muts, fill_value=0).values.T, metric='jaccard')
    sim_df = pd.DataFrame(sim, index=comps, columns=W_covspectrum.columns).fillna(0.0)

    weekly = np.asarray(result.weekly_growth_adv, dtype=float).ravel()     # 7 * f, log scale
    fitness_by_variant, best_sim = {}, {}
    for k, comp in enumerate(comps):
        variant = sim_df.loc[comp].idxmax()
        s = sim_df.loc[comp, variant]
        if s >= unassigned_threshold and s > best_sim.get(variant, -1):
            fitness_by_variant[variant], best_sim[variant] = weekly[k], s
    return fitness_by_variant, sim_df


# ── Pairwise matrices ───────────────────────────────────────────────────────

def fitnmf_growth_matrix(fitness_by_variant, variants, as_percent):
    """mat.loc[v, ref] = 7(f_v - f_ref) (log) or 100(exp(.) - 1) (%); NaN if a variant is unmatched."""
    mat = pd.DataFrame(np.nan, index=variants, columns=variants)
    for v in variants:
        for ref in variants:
            if v in fitness_by_variant and ref in fitness_by_variant:
                d = fitness_by_variant[v] - fitness_by_variant[ref]
                mat.loc[v, ref] = 100 * (np.exp(d) - 1) if as_percent else d
    return mat


def covvfit_growth_matrix(covvfit_path, variants):
    """Covvfit's Estimate (already 7(theta_v - theta_ref), log scale, per week) as a matrix."""
    fe = pd.read_csv(Path(covvfit_path) / 'pairwise_fitnesses.csv', sep='\t')
    fe.columns = [c.strip().replace('"', '') for c in fe.columns]
    for c in ('Variant', 'Reference_Variant'):
        fe[c] = fe[c].str.replace('"', '')
    mat = fe.pivot_table(index='Variant', columns='Reference_Variant', values='Estimate')
    return mat.reindex(index=variants, columns=variants)


def cocirculating_cells(variants, groups):
    """(row, col) pairs shown in the figure: row later than col in `variants`, both in one group."""
    together = {frozenset(p) for g in groups for p in combinations(g, 2)}
    return [(variants[i], variants[j]) for i in range(len(variants)) for j in range(i)
            if frozenset((variants[i], variants[j])) in together]


def paired_values(fitness_by_variant, covvfit_log, variants, groups):
    """One row per co-circulating pair with both estimates on the log scale."""
    own_log = fitnmf_growth_matrix(fitness_by_variant, variants, as_percent=False)
    rows = [dict(variant=v, reference=ref, pair=f'{v}/{ref}',
                 fitnmf=own_log.loc[v, ref], covvfit=covvfit_log.loc[v, ref])
            for v, ref in cocirculating_cells(variants, groups)]
    return pd.DataFrame(rows).dropna(subset=['fitnmf', 'covvfit']).reset_index(drop=True)


# ── Permutation test ────────────────────────────────────────────────────────

def spearman_label_permutation_test(fitness_by_variant, pairs, chunk=500_000):
    """Two-sided p-value for Spearman rho between FitNMF and Covvfit pairwise differences.
    The null permutes the FitNMF fitness values among the matched variants (all n! permutations),
    keeping Covvfit and the set of pairs fixed, because pairs that share a variant are not independent.
    p = #{|rho_perm| >= |rho_obs|} / n!  (the identity permutation is included)."""
    variants = sorted(set(pairs['variant']) | set(pairs['reference']))
    idx = {v: i for i, v in enumerate(variants)}
    values = np.array([fitness_by_variant[v] for v in variants])
    r_idx = pairs['variant'].map(idx).to_numpy()
    c_idx = pairs['reference'].map(idx).to_numpy()

    rho_obs = spearmanr(pairs['covvfit'], pairs['fitnmf'])[0]
    ry = rankdata(pairs['covvfit'].to_numpy())
    ry_c = (ry - ry.mean()) / np.sqrt(((ry - ry.mean()) ** 2).sum())

    n_ge, n_tot = 0, 0
    perms = permutations(range(len(variants)))
    while True:
        block = np.array(list(islice(perms, chunk)))
        if block.size == 0:
            break
        v = values[block]
        rx = rankdata(v[:, r_idx] - v[:, c_idx], axis=1)
        rx_c = rx - rx.mean(axis=1, keepdims=True)
        with np.errstate(invalid='ignore', divide='ignore'):
            rho = np.nan_to_num((rx_c @ ry_c) / np.sqrt((rx_c ** 2).sum(axis=1)))
        n_ge += int(np.sum(np.abs(rho) >= abs(rho_obs) - 1e-12))
        n_tot += len(block)
    assert n_tot == factorial(len(variants))
    return dict(rho=rho_obs, p_perm=n_ge / n_tot, n_permutations=n_tot,
                n_pairs=len(pairs), n_variants=len(variants))


# ── Non-overlapping point labels (panel A) ──────────────────────────────────

def _place_pair_labels(ax, xs, ys, texts, fontsize, pad_pt=1.5, marker_r_pt=5,
                       radii=(7, 15, 25, 37, 50, 65, 80, 100, 125, 150, 180, 210),
                       n_dirs=16, leader_from=12, avoid_artists=()):
    """Greedy placement: each label takes the closest free spot around its point; labels moved
    further than `leader_from` points get a thin leader line. Call after the layout is final."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ppp = fig.dpi / 72
    pad = pad_pt * ppp
    ax_box = ax.get_window_extent(renderer)

    def padded(b):
        return Bbox.from_extents(b.x0 - pad, b.y0 - pad, b.x1 + pad, b.y1 + pad)

    def overlap(a, b):
        w, h = min(a.x1, b.x1) - max(a.x0, b.x0), min(a.y1, b.y1) - max(a.y0, b.y0)
        return w * h if (w > 0 and h > 0) else 0.0

    pts = ax.transData.transform(np.column_stack([xs, ys]))
    r = marker_r_pt * ppp
    occupied = [Bbox.from_extents(px - r, py - r, px + r, py + r) for px, py in pts]
    occupied += [padded(a.get_window_extent(renderer)) for a in avoid_artists]

    base = [45, -45, 135, -135, 90, -90, 0, 180]
    extra = [a for a in np.arange(-180, 180, 360 / n_dirs) if all(abs(a - b) > 1e-6 for b in base)]
    angles = np.deg2rad(base + sorted(extra, key=lambda a: min(abs(a - b) for b in base)))
    candidates = [(rad, np.cos(a), np.sin(a)) for rad in radii for a in angles]
    outline = [pe.withStroke(linewidth=2, foreground='white')]

    def make(i, rad, cx, cy, leader):
        ha = 'left' if cx > 0.3 else 'right' if cx < -0.3 else 'center'
        va = 'bottom' if cy > 0.3 else 'top' if cy < -0.3 else 'center'
        kw = dict(xy=(xs[i], ys[i]), xytext=(rad * cx, rad * cy), textcoords='offset points',
                  ha=ha, va=va, fontsize=fontsize, annotation_clip=False)
        if leader:
            kw['arrowprops'] = dict(arrowstyle='-', color='0.3', lw=0.8, shrinkA=0, shrinkB=3,
                                    relpos=(0 if ha == 'left' else 1 if ha == 'right' else .5,
                                            0 if va == 'bottom' else 1 if va == 'top' else .5))
        t = ax.annotate(texts[i], **kw)
        t.set_path_effects(outline)
        return t

    def cost(box):
        out = (max(0, ax_box.x0 - box.x0) + max(0, box.x1 - ax_box.x1)
               + max(0, ax_box.y0 - box.y0) + max(0, box.y1 - ax_box.y1))
        return sum(overlap(box, o) for o in occupied) + out * (box.width + box.height)

    d = np.hypot(*(pts[:, None, :] - pts[None, :, :]).transpose(2, 0, 1))
    for i in np.argsort(-(d < 45 * ppp).sum(axis=1), kind='stable'):      # most crowded first
        best = None
        for rad, cx, cy in candidates:
            probe = make(i, rad, cx, cy, leader=False)
            box = padded(probe.get_window_extent(renderer))
            probe.remove()
            c = cost(box)
            if best is None or c < best[0]:
                best = (c, rad, cx, cy, box)
            if c == 0:
                break
        c, rad, cx, cy, box = best
        if c > 0:
            print(f"warning: label {texts[i]!r} overlaps; placed at the least-overlapping position")
        make(i, rad, cx, cy, leader=rad >= leader_from)
        occupied.append(box)


# ── Figure 3 ─────────────────────────────────────────────────────────────────

def make_figure3(result, data, W_covspectrum, covvfit_path, out_dir,
                 variants=VARIANTS_SC2, groups=COCIRCULATING_SC2, mutation_index=None,
                 out_stem='fig3_growth_advantages'):
    """Draws and saves Figure 3 (A: FitNMF vs. Covvfit, B: FitNMF heatmap). Returns the statistics."""
    variants = [
    'B.1.1.7', 'B.1.617.2', 'BA.1', 'BA.2', 'BA.5',
    'BQ.1.1', 'XBB.1.5', 'BA.2.75', 'EG.5',
    'JN.1'
]
    if mutation_index is None:
        mutation_index = list(next(iter(data.values())).index)
    fitness_by_variant, _ = match_fitness_to_variants(result, mutation_index, W_covspectrum)
    unmatched = [v for v in variants if v not in fitness_by_variant]
    if unmatched:
        print("no FitNMF component matched to:", unmatched)

    pairs = paired_values(fitness_by_variant, covvfit_growth_matrix(covvfit_path, variants), variants, groups)
    stats = spearman_label_permutation_test(fitness_by_variant, pairs)

    fig = plt.figure(figsize=(14, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.12], wspace=0.35)
    axA, axB = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # (A) scatter on the log scale
    x, y = pairs['covvfit'].to_numpy(), pairs['fitnmf'].to_numpy()
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    lims = [lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo)]
    axA.plot(lims, lims, ls='--', color='gray', lw=1.2, zorder=1)
    axA.scatter(x, y, s=80, color='steelblue', edgecolor='black', zorder=3)
    axA.set_xlim(lims); axA.set_ylim(lims); axA.set_aspect('equal', adjustable='box')
    axA.set_xlabel('Covvfit ' + r'$(\theta_i-\theta_j)\times 7$' + '\n(log scale, per week)', fontsize=FS_LABEL)
    axA.set_ylabel('FitNMF ' + r'$(f_i-f_j)\times 7$' + '\n(log scale, per week)', fontsize=FS_LABEL)
    axA.tick_params(labelsize=FS_TICK)
    stat_txt = axA.text(0.04, 0.96, f"Spearman ρ = {stats['rho']:.2f}\npermutation p = {stats['p_perm']:.2g}",
                        transform=axA.transAxes, ha='left', va='top', fontsize=FS_STAT)

    # (B) FitNMF heatmap in %, co-circulating pairs only, lower triangle without the empty row/column
    variants = [
        'B.1.1.7', 'B.1.617.2', 'BA.1', 'BA.2', 'BA.5',
        'BQ.1.1', 'BA.2.75', 'XBB.1.5', 'EG.5',
        'JN.1'
    ]
    own_pct = fitnmf_growth_matrix(fitness_by_variant, variants, as_percent=True)
    shown = np.zeros(own_pct.shape, dtype=bool)
    for v, ref in cocirculating_cells(variants, groups):
        shown[variants.index(v), variants.index(ref)] = True
    sub = own_pct.where(shown).iloc[1:, :-1]
    vals = sub.to_numpy()
    has_neg = np.nanmin(vals) < 0                     # negative cells get the lightest color; the
    sns.heatmap(sub, ax=axB, cmap='Reds', vmin=0, vmax=np.nanmax(vals),   # number shows the sign
                annot=True, fmt='.0f', annot_kws={'size': FS_HM_ANNOT},
                linewidths=0.5, linecolor='gray',
                cbar_kws={'label': 'Weekly growth advantage (%)', 'shrink': 0.85,
                          'extend': 'min' if has_neg else 'neither'})
    cbar = axB.collections[0].colorbar
    cbar.ax.tick_params(labelsize=FS_HM_CBAR)
    cbar.ax.yaxis.label.set_size(FS_HM_LABEL - 2)
    axB.set_xlabel('Reference variant', fontsize=FS_HM_LABEL)
    axB.set_ylabel('Variant', fontsize=FS_HM_LABEL)
    plt.setp(axB.get_xticklabels(), rotation=45, ha='right', fontsize=FS_HM_TICK)
    plt.setp(axB.get_yticklabels(), rotation=0, fontsize=FS_HM_TICK)

    for ax, letter in ((axA, 'A'), (axB, 'B')):
        ax.text(-0.22, 1.04, letter, transform=ax.transAxes, fontsize=FS_PANEL, fontweight='bold',
                ha='left', va='bottom')

    _place_pair_labels(axA, x, y, pairs['pair'].tolist(), fontsize=FS_PAIR, avoid_artists=[stat_txt])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(out_dir / f'{out_stem}.{ext}', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"{stats['n_pairs']} co-circulating pairs, {stats['n_variants']} matched variants: "
          f"Spearman rho = {stats['rho']:.3f}, two-sided permutation p = {stats['p_perm']:.2g} "
          f"({stats['n_permutations']:,} permutations)")
    return stats