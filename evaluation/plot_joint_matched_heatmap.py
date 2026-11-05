"""
plot_joint_matched_heatmap.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
from scipy.spatial.distance import cdist
import matplotlib.colors as mcolors

# ── Nature-style figure conventions ──────────────────────────────────────────
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

LINESTYLES = ['-']


def replicator_P(f, b, t):
    """Parametric (replicator) abundances P_k(t) = exp(f_k t + b_k) / sum_i exp(f_i t + b_i).

    f, b : length-k arrays (fitness and offset per component; the reference
           component has f = b = 0 by convention).
    t    : length-T array of times. It MUST be the same time axis (units and
           origin) that the fit used, otherwise b is applied at the wrong t.
    Returns a (k, T) array whose columns sum to 1.
    """
    f = np.asarray(f, dtype=float).reshape(-1, 1)
    b = np.asarray(b, dtype=float).reshape(-1, 1)
    t = np.asarray(t, dtype=float).reshape(1, -1)
    z = f * t + b
    z = z - z.max(axis=0, keepdims=True)          # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=0, keepdims=True)


def _match_components_to_ground_truth(W_binary,
                                       W_covspectrum,
                                       unassigned_threshold=0.5,
                                       unassigned_label='unassigned',
                                       tie_tol=1e-9,
                                       variant_colors=None):
    """Many-to-one, per-component match of predicted components to ground-truth
    CovSpectrum variants via Jaccard similarity on binarized W.

    Every variant within `tie_tol` of the best similarity is reported, not just
    the argmax -- argmax silently returns whichever tied column comes first,
    which hides genuine ties (common here, since 'BA.5*' in the CovSpectrum
    query includes BQ.1.1, so those two columns are near-duplicates).

    unassigned_threshold prevents a weak/spurious match from being displayed
    with the same confidence as a genuine one.

    Returns (sim_df, new_labels, best_labels, best_matches), where
    best_matches[k] is the LIST of tied variant names for component k -- needed
    by _reorder, which can't parse the joined display string.
    """
    all_muts = sorted(set(W_binary.index) | set(W_covspectrum.index))
    W_pred_final = W_binary.reindex(all_muts, fill_value=0)
    W_true_final = W_covspectrum.reindex(all_muts, fill_value=0)

    dist_matrix = cdist(W_pred_final.values.T, W_true_final.values.T, metric='jaccard')
    sim_matrix = 1 - dist_matrix

    sim_df = pd.DataFrame(sim_matrix, index=W_binary.columns, columns=W_covspectrum.columns)
    sim_df = sim_df.fillna(0.0)

    best_sim = sim_df.values.max(axis=1)
    tied = [list(sim_df.columns[np.flatnonzero(row >= s - tie_tol)])
            for row, s in zip(sim_df.values, best_sim)]

    best_labels, best_matches = [], []
    for variants, s in zip(tied, best_sim):
        if s >= unassigned_threshold:
            best_labels.append(' / '.join(variants))
            best_matches.append(variants)
        else:
            best_labels.append(unassigned_label)
            best_matches.append([])

    counts = {}
    new_labels = []
    for lbl in best_labels:
        counts[lbl] = counts.get(lbl, 0) + 1
        new_labels.append(lbl if counts[lbl] == 1 else f"{lbl} ({counts[lbl]})")

    return sim_df, new_labels, best_labels, best_matches


def _reorder(sim_df, new_labels, best_matches, covspectrum_order):
    """Reorder rows/cols so matched pairs sit together, unmatched trail after.
    Uses best_matches (the variant lists) rather than the joined labels, so a
    tied component is ordered next to its first matched variant and ALL of its
    tied variants count as matched columns."""
    gt_to_pred_labels = {}
    for pred_label, variants in zip(new_labels, best_matches):
        if variants:
            gt_to_pred_labels.setdefault(variants[0], []).append(pred_label)

    row_order_matched = []
    for gt in covspectrum_order:
        if gt in gt_to_pred_labels:
            row_order_matched.extend(gt_to_pred_labels[gt])
    row_order_unmatched = [c for c in new_labels if c not in set(row_order_matched)]
    row_order = row_order_matched + row_order_unmatched

    matched_gt = {v for variants in best_matches for v in variants}
    col_matched = [c for c in covspectrum_order if c in matched_gt]
    col_unmatched = [c for c in covspectrum_order if c not in matched_gt]
    col_order = col_matched + col_unmatched

    return sim_df.loc[row_order, col_order], row_order


def plot_matched_heatmap_and_abundances(
        result, data, W_covspectrum,
        mutation_index=None,
        P=None,
        binarize_threshold=0.5,
        unassigned_threshold=0.5,
        tie_tol=1e-9,
        variant_colors=None,
        p_darken=0.6,
        p_linewidth=1.2,
        panel_width=18, heatmap_height=4, abundance_height=2.5,
        title_fontsize=18,
        label_fontsize=13,
        tick_fontsize=13,
        annot_fontsize=14,
        peak_label_fontsize=14,
        legend_fontsize=14,
        out_path='results_joint_nmf.png'):
    """One shared Jaccard heatmap on top (W is shared across locations in the
    joint model), one shared abundance axes below with every location
    overlaid -- color = matched component, linestyle = location.

    Solid lines are the estimated abundances H. If P is available, the
    parametric (replicator) abundances P(f, b; t) are overlaid as dashed lines
    of a darker shade of the same colour.

    P : dict {location: (k, T) array} aligned with result.H[location] (same
        component order, same time points). If None, result.P is used when it
        exists; otherwise no dashed lines are drawn. Use replicator_P(f, b, t)
        to build it from fitted f, b and the time axis of the fit.

    Components whose best Jaccard ties across several ground-truth variants are
    labelled with all of them (see tie_tol)."""
    locations = list(result.H.keys())
    rank = result.W.shape[1]

    if mutation_index is None:
        mutation_index = list(data[locations[0]].index)
    if len(mutation_index) != result.W.shape[0]:
        raise ValueError(
            f"mutation_index has {len(mutation_index)} entries but result.W has "
            f"{result.W.shape[0]} rows -- pass the correct mutation_index explicitly.")

    # ── Parametric abundances P (optional) ──────────────────────────────────
    if P is None:
        P = getattr(result, 'P', None)
    if P is not None:
        missing = [loc for loc in locations if loc not in P]
        if missing:
            raise KeyError(f"P has no entry for location(s): {missing}")
        for loc in locations:
            if np.shape(P[loc]) != np.shape(result.H[loc]):
                raise ValueError(
                    f"P[{loc!r}] has shape {np.shape(P[loc])} but result.H[{loc!r}] "
                    f"has shape {np.shape(result.H[loc])} -- P must be (k, T) and "
                    f"aligned with H (same component order and time points).")
    else:
        print("plot_matched_heatmap_and_abundances: no P given (and result.P not "
              "found) -- dashed parametric lines are not drawn.")

    original_cols = [f'Component_{k}' for k in range(rank)]
    W_df = pd.DataFrame(result.W, index=mutation_index, columns=original_cols)
    W_binary = (W_df >= binarize_threshold).astype(int)

    sim_df, new_labels, best_labels, best_matches = _match_components_to_ground_truth(
        W_binary, W_covspectrum,
        unassigned_threshold=unassigned_threshold, tie_tol=tie_tol)
    sim_df.index = new_labels

    covspectrum_order = list(W_covspectrum.columns)
    sim_df, row_order = _reorder(sim_df, new_labels, best_matches, covspectrum_order)

    label_to_k = {label: k for k, label in enumerate(new_labels)}

    DEFAULT_VARIANT_COLORS = {
    'B.1.1.7': '#1f77b4', 'B.1.617.2': '#ff7f0e', 'BA.1': '#98df8a', 'BA.2': '#ff9896',
    'BA.5': '#8c564b', 'BQ.1.1': '#e377c2', 'XBB.1.5': 'dimgrey', 'EG.5': '#dbdb8d',
    'JN.1': '#9edae5', 'BA.2.75': '#c49c94'}


    variant_colors = DEFAULT_VARIANT_COLORS if variant_colors is None else variant_colors
    hexof = mcolors.to_hex
    reserved = {hexof(c) for c in variant_colors.values()}   # never reused as fallback
    greys = {'#c7c7c7', '#7f7f7f'}
    pool = [c for c in plt.cm.tab20.colors if hexof(c) not in reserved | greys]

    color_of, used = {}, set()
    for label in row_order:                                   # pass 1: known variants
        matched = best_matches[label_to_k[label]]             # [] if unassigned
        col = variant_colors.get(matched[0]) if matched else None
        if col is not None and hexof(col) not in used:
            color_of[label] = col
            used.add(hexof(col))
    for label in row_order:                                   # pass 2: unknown / duplicates
        if label not in color_of:
            col = next((c for c in pool if hexof(c) not in used), 'black')
            color_of[label] = col
            used.add(hexof(col))
    colors = [color_of[label] for label in row_order]

    fig = plt.figure(figsize=(panel_width, heatmap_height + abundance_height))
    gs = fig.add_gridspec(2, 1, height_ratios=[heatmap_height, abundance_height], hspace=0.6)

    # ── Heatmap (shared across all locations -- W is shared) ────────────────
    ax_heat = fig.add_subplot(gs[0])
    sns.heatmap(sim_df, ax=ax_heat, cmap='Blues', vmin=0, vmax=1,
                annot=True, fmt='.2f',
                annot_kws={'size': annot_fontsize},
                cbar_kws={'label': 'Jaccard similarity', 'shrink': 0.85})
    #ax_heat.set_title('Jaccard similarity — predicted vs CovSpectrum (many-to-one)',
                      # fontsize=title_fontsize, fontweight='bold', pad=14)
    ax_heat.set_xlabel('CovSpectrum variant', fontsize=label_fontsize)
    ax_heat.set_ylabel('Predicted component', fontsize=label_fontsize)
    ax_heat.tick_params(axis='x', rotation=45, labelsize=tick_fontsize)
    ax_heat.tick_params(axis='y', rotation=0, labelsize=tick_fontsize)
    for tick in ax_heat.get_xticklabels():
        tick.set_ha('right')

    cbar = ax_heat.collections[0].colorbar
    cbar.ax.tick_params(labelsize=tick_fontsize)
    cbar.set_label('Jaccard similarity', fontsize=label_fontsize)

    # ── One shared abundance axes, all locations overlaid ────────────────────
    from src.helpers import _parse_dates  # local import to avoid a hard module-load-order dependency

    ax = fig.add_subplot(gs[1])
    linestyles = [LINESTYLES[i % len(LINESTYLES)] for i in range(len(locations))]
    best_peak_value = {label: -np.inf for label in row_order}
    best_peak_date = {label: None for label in row_order}

    for loc, ls in zip(locations, linestyles):
        _, dates = _parse_dates(data[loc])
        dates = pd.to_datetime(dates)
        H_loc = result.H[loc]
        P_loc = np.asarray(P[loc]) if P is not None else None

        for pos, label in enumerate(row_order):
            k = label_to_k[label]
            color = colors[pos]
            values = H_loc[k, :]
            ax.plot(dates, values, color=color, linestyle=ls, linewidth=1.6, alpha=0.7)

            if P_loc is not None:
                # darker shade of the component colour so the dashes stay visible
                # where P and H coincide
                p_color = tuple(p_darken * c for c in mcolors.to_rgb(color))
                ax.plot(dates, P_loc[k, :], color=p_color, linestyle='--',
                        linewidth=p_linewidth, zorder=3)

            peak_idx = np.argmax(values)
            if values[peak_idx] > best_peak_value[label]:
                best_peak_value[label] = values[peak_idx]
                best_peak_date[label] = dates[peak_idx]

    for pos, label in enumerate(row_order):
        if best_peak_date[label] is None:
            continue
        # tied variants stack vertically above the curve; the heatmap row label
        # keeps the ' / ' form, where vertical space is tighter
        txt = ax.annotate(label.replace(' / ', '\n'),
                           xy=(best_peak_date[label], best_peak_value[label]),
                           xytext=(0, 6), textcoords='offset points',
                           ha='center', va='bottom',
                           fontsize=peak_label_fontsize, color=colors[pos], fontweight='bold')
        txt.set_path_effects([pe.withStroke(linewidth=2.2, foreground='white')])

    ax.set_xlabel('Date', fontsize=label_fontsize)
    ax.set_ylabel('Relative abundance', fontsize=label_fontsize)
    #ax.set_title('Variant abundance trajectories (matched components)',
                 #fontsize=title_fontsize, fontweight='bold', pad=14)
    ax.tick_params(axis='x', rotation=45, labelsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.spines[['top', 'right']].set_visible(False)

    loc_handles = [plt.Line2D([0], [0], color='black', linestyle=ls, linewidth=1.6)
                   for ls in linestyles]
    # The location legend only carries information if locations differ in linestyle
    # (with LINESTYLES = ['-'] all entries look identical). It is kept when P is
    # not drawn, to preserve the original behaviour.
    loc_legend_informative = len(set(linestyles)) > 1
    if P is None or loc_legend_informative:
        leg_loc = ax.legend(loc_handles, locations, title='Location', loc='upper right',
                            fontsize=legend_fontsize, title_fontsize=legend_fontsize, frameon=False)

    if P is not None:
        # solid = estimated H, dashed = parametric P
        if loc_legend_informative:
            ax.add_artist(leg_loc)
        type_handles = [plt.Line2D([0], [0], color='black', linestyle='-', linewidth=1.6),
                        plt.Line2D([0], [0], color='black', linestyle='--', linewidth=p_linewidth)]
        # placed just above the axes (top right) so it never covers a curve
        ax.legend(type_handles, ['Estimated abundance $H$', 'Parametric model $P$'],
                  loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=2,
                  fontsize=legend_fontsize, frameon=False)

    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    if out_path.lower().endswith('.png'):
        fig.savefig(out_path[:-4] + '.pdf', bbox_inches='tight')
    return fig