"""
Plotting utilities for iterative incompatibility scoring results.

Three main visualisations:

1. winner_heatmap        – single figure; x = genomic position, y = iteration,
                           colour = which node dominates at each split‐point region.

2. peeling_summary       – compact panel showing only the iterations where the
                           identity of the winning node changes, i.e. the moments
                           a new recombination signal is "uncovered".

3. score_trajectories    – one line per node showing its total score across
                           iterations.  Peaks and drops reveal when each
                           recombination's sites are peeled away.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.collections import PatchCollection


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_positions(iterations, site_positions=None):
    """
    Return a function that maps a global site index → x-coordinate.

    If *site_positions* is provided (array where site_positions[i] is the
    genomic position of site i), use that.  Otherwise fall back to the
    raw site index.
    """
    if site_positions is not None:
        return lambda idx: site_positions[idx]
    return lambda idx: idx


def _collect_winners(iterations):
    """
    For every iteration and every split-point region, find the node with
    the highest score.

    Returns
    -------
    list of list of dict
        Outer list = iterations.  Inner list = split-point regions.
        Each dict: {'left': global_site_left,
                    'right': global_site_right,
                    'winner': node_index,
                    'score': float}
    """
    all_winners = []
    for it in iterations:
        sites = it['incomp_sites']
        scores = it['scores']              # (n_split, n_nodes)
        regions = []
        for k in range(scores.shape[0]):
            winner = int(np.argmax(scores[k]))
            regions.append({
                'left': int(sites[k]),
                'right': int(sites[k + 1]),
                'winner': winner,
                'score': float(scores[k, winner]),
            })
        all_winners.append(regions)
    return all_winners


def _unique_winners(all_winners):
    """Set of all node ids that are ever a winner."""
    nodes = set()
    for regions in all_winners:
        for r in regions:
            nodes.add(r['winner'])
    return sorted(nodes)


def _make_node_cmap(winner_nodes):
    """
    Build a colormap + normaliser that maps each winner node id to a
    distinct colour.  Returns (cmap, norm, node_to_colour_idx).
    """
    n = len(winner_nodes)
    base = plt.cm.get_cmap('tab20', max(n, 2))
    colours = [base(i) for i in range(n)]
    cmap = ListedColormap(colours)
    boundaries = np.arange(n + 1) - 0.5
    norm = BoundaryNorm(boundaries, cmap.N)
    node_to_cidx = {node: i for i, node in enumerate(winner_nodes)}
    return cmap, norm, node_to_cidx


# ─── 1.  Winner heatmap ──────────────────────────────────────────────────────

def winner_heatmap(iterations, site_positions=None, ax=None, figsize=(14, 8),
                   title='Winner heatmap'):
    """
    Single figure.
    X-axis = genomic position (or site index).
    Y-axis = iteration (top = first / most sites, bottom = fewest).
    Colour = which node has the highest score in that split-point region.

    Parameters
    ----------
    iterations : list of dict
        Output of ``iterative_removal_scoring``.
    site_positions : array-like, optional
        Maps global site index → genomic position.  If *None*, raw site
        indices are used on the x-axis.
    ax : matplotlib Axes, optional
    figsize : tuple
    title : str

    Returns
    -------
    fig, ax
    """
    if not iterations:
        raise ValueError('No iterations to plot.')

    pos = _get_positions(iterations, site_positions)
    all_winners = _collect_winners(iterations)
    winner_nodes = _unique_winners(all_winners)
    cmap, norm, node2c = _make_node_cmap(winner_nodes)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    n_iter = len(iterations)
    patches = []
    colours = []

    for row, regions in enumerate(all_winners):
        y_bottom = n_iter - 1 - row          # top = first iteration
        for r in regions:
            x0 = pos(r['left'])
            x1 = pos(r['right'])
            rect = mpatches.Rectangle((x0, y_bottom), x1 - x0, 1)
            patches.append(rect)
            colours.append(node2c[r['winner']])

    col = PatchCollection(patches, cmap=cmap, norm=norm,
                          edgecolor='white', linewidth=0.3)
    col.set_array(np.array(colours, dtype=float))
    ax.add_collection(col)

    # axis limits
    all_sites = iterations[0]['incomp_sites']
    x_min = pos(int(all_sites[0]))
    x_max = pos(int(all_sites[-1]))
    margin = (x_max - x_min) * 0.01
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(0, n_iter)

    # y-tick labels:  "iter 0 (N sites)" at top, descending
    yticks = np.arange(n_iter)
    ylabels = [f'{iterations[n_iter - 1 - i]["n_incomp"]} sites'
               for i in range(n_iter)]
    # thin out labels if many iterations
    if n_iter > 30:
        step = max(n_iter // 15, 1)
        yticks = yticks[::step]
        ylabels = ylabels[::step]
    ax.set_yticks(yticks + 0.5)
    ax.set_yticklabels(ylabels, fontsize=7)

    ax.set_xlabel('Genomic position' if site_positions is not None else 'Site index')
    ax.set_ylabel('Iteration  (top = most sites)')
    ax.set_title(title)

    # legend
    handles = [mpatches.Patch(color=cmap(node2c[n]),
                              label=f'node {n}')
               for n in winner_nodes]
    # place outside if many nodes
    if len(handles) <= 15:
        ax.legend(handles=handles, loc='center left',
                  bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False)
    else:
        ax.legend(handles=handles[:15], loc='center left',
                  bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False,
                  title=f'{len(handles)} nodes (showing 15)')

    fig.tight_layout()
    return fig, ax


# ─── 3.  Peeling summary ─────────────────────────────────────────────────────

def peeling_summary(iterations, site_positions=None, figsize=(14, 6),
                    title='Peeling summary – recombination signals uncovered'):
    """
    Compact view showing only the iterations at which the *identity* of the
    winning node changes somewhere along the sequence.  Each selected
    iteration is drawn as a row of coloured blocks (like the heatmap),
    but only "transition" iterations are kept.

    An additional top‐panel bar chart shows the removed site's mutation
    time, giving a sense of the chronological order of peeling.

    Parameters
    ----------
    iterations : list of dict
    site_positions : array-like, optional
    figsize : tuple
    title : str

    Returns
    -------
    fig, axes   (axes is a length-2 array: [time_bar, heatmap])
    """
    if not iterations:
        raise ValueError('No iterations to plot.')

    pos = _get_positions(iterations, site_positions)
    all_winners = _collect_winners(iterations)
    winner_nodes = _unique_winners(all_winners)
    cmap, norm, node2c = _make_node_cmap(winner_nodes)

    # ── detect transition iterations ──
    # Build a "winner signature" per iteration: tuple of winners at each
    # split point.  We keep an iteration if its signature differs from the
    # previous one (plus always keep the first).
    def _signature(regions):
        return tuple(r['winner'] for r in regions)

    keep_idx = [0]
    prev_sig = _signature(all_winners[0])
    for i in range(1, len(all_winners)):
        sig = _signature(all_winners[i])
        if sig != prev_sig:
            keep_idx.append(i)
            prev_sig = sig

    n_keep = len(keep_idx)

    fig, axes = plt.subplots(2, 1, figsize=figsize,
                             gridspec_kw={'height_ratios': [1, max(n_keep, 2)]},
                             sharex=True)
    ax_time, ax_heat = axes

    # ── time bar (top) ──
    for i in keep_idx:
        it = iterations[i]
        removed_pos = pos(it['removed_site'])
        ax_time.bar(removed_pos, it['removed_time'],
                    width=(pos(int(iterations[0]['incomp_sites'][-1])) -
                           pos(int(iterations[0]['incomp_sites'][0]))) * 0.008,
                    color='0.35', zorder=3)
    ax_time.set_ylabel('Removed\nmut. time', fontsize=8)
    ax_time.tick_params(labelsize=7)
    ax_time.set_title(title, fontsize=10)

    # ── heatmap rows (bottom) ──
    patches = []
    colours = []

    for row_i, iter_i in enumerate(keep_idx):
        regions = all_winners[iter_i]
        y_bottom = n_keep - 1 - row_i
        for r in regions:
            x0 = pos(r['left'])
            x1 = pos(r['right'])
            rect = mpatches.Rectangle((x0, y_bottom), x1 - x0, 1)
            patches.append(rect)
            colours.append(node2c[r['winner']])

    col = PatchCollection(patches, cmap=cmap, norm=norm,
                          edgecolor='white', linewidth=0.4)
    col.set_array(np.array(colours, dtype=float))
    ax_heat.add_collection(col)

    all_sites = iterations[0]['incomp_sites']
    x_min = pos(int(all_sites[0]))
    x_max = pos(int(all_sites[-1]))
    margin = (x_max - x_min) * 0.01
    ax_heat.set_xlim(x_min - margin, x_max + margin)
    ax_heat.set_ylim(0, n_keep)

    yticks = np.arange(n_keep)
    ylabels = [f'iter {keep_idx[n_keep - 1 - i]}  '
               f'({iterations[keep_idx[n_keep - 1 - i]]["n_incomp"]} sites)'
               for i in range(n_keep)]
    ax_heat.set_yticks(yticks + 0.5)
    ax_heat.set_yticklabels(ylabels, fontsize=7)
    ax_heat.set_xlabel('Genomic position' if site_positions is not None
                       else 'Site index')

    # legend
    handles = [mpatches.Patch(color=cmap(node2c[n]), label=f'node {n}')
               for n in winner_nodes]
    ax_heat.legend(handles=handles[:20], loc='center left',
                   bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False)

    fig.tight_layout()
    return fig, axes


# ─── 4.  Score trajectories ──────────────────────────────────────────────────

def score_trajectories(iterations, top_n=10, figsize=(12, 5),
                       title='Node score trajectories across iterations'):
    """
    Line plot: x = iteration, y = total score (summed across split points).
    One line per node, showing only the *top_n* most prominent nodes
    (ranked by maximum total score across all iterations).

    Peaks reveal which recombination a node is associated with; drops show
    the moment its sites are peeled away.

    Parameters
    ----------
    iterations : list of dict
    top_n : int
        Number of nodes to display.
    figsize : tuple
    title : str

    Returns
    -------
    fig, ax
    """
    if not iterations:
        raise ValueError('No iterations to plot.')

    n_nodes = iterations[0]['total'].shape[0]
    n_iter = len(iterations)

    # build (n_iter, n_nodes) matrix of total scores
    totals = np.zeros((n_iter, n_nodes))
    for i, it in enumerate(iterations):
        totals[i] = it['total']

    # rank nodes by their peak total score
    peak_scores = totals.max(axis=0)
    top_nodes = np.argsort(peak_scores)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.cm.get_cmap('tab20', max(top_n, 2))
    x = np.arange(n_iter)

    for rank, node in enumerate(top_nodes):
        ax.plot(x, totals[:, node], color=cmap(rank), linewidth=1.5,
                label=f'node {node}', alpha=0.85)

    # mark removed-site times as a secondary axis
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total score (summed across split points)')
    ax.set_title(title)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),
              fontsize=8, frameon=False)

    # add removed-site annotation along the top
    ax2 = ax.twiny()
    removed_times = [it['removed_time'] for it in iterations]
    ax2.set_xlim(ax.get_xlim())
    # thin out tick labels
    step = max(n_iter // 12, 1)
    tick_pos = x[::step]
    tick_labels = [f'{removed_times[i]:.0f}' for i in tick_pos]
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(tick_labels, fontsize=7, rotation=45)
    ax2.set_xlabel('Removed mutation time', fontsize=8)

    fig.tight_layout()
    return fig, ax


# ─── convenience: all three in one call ──────────────────────────────────────

def plot_all(iterations, site_positions=None, top_n=10, save_prefix=None):
    """
    Generate all three plots.  If *save_prefix* is given, save each as
    ``{save_prefix}_heatmap.pdf``, ``{save_prefix}_peeling.pdf``, and
    ``{save_prefix}_trajectories.pdf``.

    Returns
    -------
    dict of (fig, axes) keyed by 'heatmap', 'peeling', 'trajectories'
    """
    results = {}

    #fig1, ax1 = winner_heatmap(iterations, site_positions=site_positions)
    #results['heatmap'] = (fig1, ax1)

    # fig2, ax2 = peeling_summary(iterations, site_positions=site_positions)
    # results['peeling'] = (fig2, ax2)

    fig3, ax3 = score_trajectories(iterations, top_n=top_n)
    results['trajectories'] = (fig3, ax3)

    if save_prefix:
        #fig1.savefig(f'{save_prefix}_heatmap.pdf', bbox_inches='tight')
        # fig2.savefig(f'{save_prefix}_peeling.pdf', bbox_inches='tight')
        fig3.savefig(f'{save_prefix}_trajectories.pdf', bbox_inches='tight')

    return results
