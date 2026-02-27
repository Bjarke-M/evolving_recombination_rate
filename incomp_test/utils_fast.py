import tskit
import msprime
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# ─── Incompatibility matrix (vectorized via dot products) ─────────────────────

def compute_incompatibility_matrix_2(genotype_matrix, recsites=set()):
    """
    Computes the 4-gamete test incompatibility matrix using matrix dot products.
    """
    G = genotype_matrix.astype(np.float32)
    G_inv = 1.0 - G

    has_11 = (G @ G.T) > 0
    has_10 = (G @ G_inv.T) > 0
    has_01 = (G_inv @ G.T) > 0
    has_00 = (G_inv @ G_inv.T) > 0

    incomp = has_11 & has_10 & has_01 & has_00
    matrix = incomp.astype(np.int8)
    np.fill_diagonal(matrix, 0)

    if recsites:
        rec_list = list(recsites)
        matrix[rec_list, :] = 0
        matrix[:, rec_list] = 0

    return matrix


# ─── Pair genotype computation (vectorized) ───────────────────────────────────

def incomp_pair_genotypes(incomp_genotypes, pair_matrix):
    """
    Vectorized pair genotype computation.

    Parameters
    ----------
    incomp_genotypes : (n_nodes, n_incomp_sites) array
    pair_matrix : (n_incomp_sites, n_incomp_sites) array

    Returns
    -------
    results : (n_pairs, n_nodes) int8 array, values 0-3
    pairs : (n_pairs, 2) int array
    """
    i_idx, j_idx = np.where(np.triu(pair_matrix, k=1))

    if len(i_idx) == 0:
        return np.empty((0, incomp_genotypes.shape[0]), dtype=np.int8), np.empty((0, 2), dtype=np.intp)

    g_i = incomp_genotypes[:, i_idx].T
    g_j = incomp_genotypes[:, j_idx].T

    results = (g_i | (g_j << 1)).astype(np.int8)
    pairs = np.column_stack((i_idx, j_idx))

    return results, pairs


# ─── Scoring functions ────────────────────────────────────────────────────────

def score_pairs_matrix(cross_genotypes, w=1.0):
    """
    Score each pair independently. Returns (n_pairs, n_nodes).

    For each pair/individual:
      +1 if genotype == 3
      +w if individual is the sole carrier of its genotype in that pair
    """
    n_pairs, n_ind = cross_genotypes.shape
    scores = np.zeros((n_pairs, n_ind), dtype=np.float32)

    if n_pairs == 0:
        return scores

    scores[cross_genotypes == 3] += 1.0

    for v in range(4):
        is_v = (cross_genotypes == v)
        v_counts = is_v.sum(axis=1)
        unique_v_mask = (v_counts == 1)
        scores += (is_v & unique_v_mask[:, None]) * w

    return scores



def iterative_removal_scoring(genotype_matrix, incomp_matrix, site_mut_times, w=1):
    """
    Iteratively remove the youngest incompatible site, re-score, and collect
    results — using INCREMENTAL UPDATES instead of full recomputation.

    When a site is removed, only the pairs involving that site are affected.
    We subtract their contribution from the diff array and recompute the
    cumulative sum (which is cheap: O(n_incomp × n_nodes)).

    The full pair scoring (the expensive part) is done only ONCE.
    """
    current_matrix = incomp_matrix.copy()
    iterations = []

    incomp_sites = np.where(current_matrix.any(axis=1))[0]
    n_initial = len(incomp_sites)

    if n_initial < 2:
        return iterations

    n_nodes = genotype_matrix.shape[1]

    # ── One-time full computation ──
    incomp_geno = genotype_matrix[incomp_sites, :].T
    sub_incomp = current_matrix[np.ix_(incomp_sites, incomp_sites)]
    all_genos, pairs = incomp_pair_genotypes(incomp_geno, sub_incomp)

    if len(pairs) == 0:
        return iterations

    # Score all pairs once (the expensive step — done only once!)
    pair_scores = score_pairs_matrix(all_genos, w=w)

    n_incomp = len(incomp_sites)

    # Build the initial difference array
    diff_array = np.zeros((n_incomp, n_nodes), dtype=np.float32)
    np.add.at(diff_array, pairs[:, 0], pair_scores)
    np.add.at(diff_array, pairs[:, 1], -pair_scores)

    # Track which pairs are still alive
    alive = np.ones(len(pairs), dtype=bool)

    # Map global site index → local index in incomp_sites
    global_to_local = {int(s): i for i, s in enumerate(incomp_sites)}

    # Track which local sites are still active (not yet removed)
    active_locals = np.ones(n_incomp, dtype=bool)

    pbar = tqdm(total=max(n_initial - 1, 0), desc="Iterative scoring")

    while True:
        active_global = incomp_sites[active_locals]

        if len(active_global) < 2:
            break

        # Compute score matrix from diff array (cheap cumsum)
        cumsum_full = np.cumsum(diff_array, axis=0)

        # Extract scores at split points between consecutive active sites
        active_local_indices = np.where(active_locals)[0]
        score_mat = cumsum_full[active_local_indices[:-1], :]

        # Identify youngest site among active
        youngest_global = active_global[np.argmin(site_mut_times[active_global])]
        youngest_time = site_mut_times[youngest_global]
        youngest_local = global_to_local[int(youngest_global)]

        iterations.append({
            'incomp_sites': active_global.copy(),
            'scores': score_mat.copy(),
            'total': score_mat.sum(axis=0),
            'removed_site': int(youngest_global),
            'removed_time': float(youngest_time),
            'n_incomp': len(active_global),
        })

        # ── Incremental update: subtract pairs involving the removed site ──
        affected_mask = alive & (
            (pairs[:, 0] == youngest_local) | (pairs[:, 1] == youngest_local)
        )
        affected_indices = np.where(affected_mask)[0]

        if len(affected_indices) > 0:
            aff_pairs = pairs[affected_indices]
            aff_scores = pair_scores[affected_indices]

            np.add.at(diff_array, aff_pairs[:, 0], -aff_scores)
            np.add.at(diff_array, aff_pairs[:, 1], aff_scores)

            alive[affected_mask] = False

        active_locals[youngest_local] = False

        pbar.update(1)

    pbar.close()
    return iterations


# ─── NON-INCREMENTAL version (for correctness checking / fallback) ────────────

def iterative_removal_scoring_full(genotype_matrix, incomp_matrix, site_mut_times, w=1):
    """
    Original full-recomputation version (with fast vectorized internals).
    Use this for correctness validation against the incremental version.
    """
    current_matrix = incomp_matrix.copy()
    iterations = []

    n_initial = np.count_nonzero(current_matrix.any(axis=1))
    pbar = tqdm(total=max(n_initial - 1, 0), desc="Iterative scoring (full)")

    while True:
        incomp_sites = np.where(current_matrix.any(axis=1))[0]

        if len(incomp_sites) < 2:
            break

        youngest_site = incomp_sites[np.argmin(site_mut_times[incomp_sites])]
        youngest_time = site_mut_times[youngest_site]

        result = score_at_split_points(genotype_matrix, current_matrix, w=w)

        iterations.append({
            'incomp_sites': result['incomp_sites'],
            'scores': result['scores'],
            'total': result['scores'].sum(axis=0),
            'removed_site': int(youngest_site),
            'removed_time': float(youngest_time),
            'n_incomp': len(incomp_sites),
        })

        current_matrix[youngest_site, :] = 0
        current_matrix[:, youngest_site] = 0
        pbar.update(1)

    pbar.close()
    return iterations


# ─── Recombination event summary functions (unchanged) ────────────────────────

def find_recombination_events(ts):
    events = []
    visited = set()

    for node_id in range(ts.num_nodes):
        if node_id in visited:
            continue
        node = ts.node(node_id)
        if node.flags != msprime.NODE_IS_RE_EVENT:
            continue
        partner = node_id + 1
        node2 = ts.node(partner)
        if node2.flags != msprime.NODE_IS_RE_EVENT or node2.time != node.time:
            continue
        visited.update([node_id, partner])

        edges_left = [e for e in ts.edges() if e.parent == node_id]
        edges_right = [e for e in ts.edges() if e.parent == partner]

        if edges_left and edges_right:
            child = edges_left[0].child
            breakpoint_ = min(e.right for e in edges_left)
            events.append({
                'rec_node_left': node_id,
                'rec_node_right': partner,
                'child': child,
                'breakpoint': breakpoint_,
                'time': node.time,
            })

    return events


def get_descendant_samples(ts, rec_events):
    for event in rec_events:
        tree = ts.at(event['breakpoint'] - 1)
        event['descendant_samples'] = sorted(tree.samples(event['child']))
        event['descendant_nodes'] = sorted(
            n for n in tree.nodes(event['child'])
            if n != event['child'] and ts.node(n).flags != msprime.NODE_IS_RE_EVENT
        )
    return rec_events


def get_tree_indices(ts, rec_events):
    for event in rec_events:
        bp = event['breakpoint']
        tree_left = ts.at(bp - 1)
        tree_right = ts.at(bp)
        event['tree_index_left'] = tree_left.index
        event['tree_index_right'] = tree_right.index
    return rec_events


def count_coalescents(ts, rec_events):
    for event in rec_events:
        node = event['child']
        tree_left = ts.at_index(event['tree_index_left'])
        tree_right = ts.at_index(event['tree_index_right'])

        path_left = []
        curr = node
        while curr != tskit.NULL:
            if ts.node(curr).flags != msprime.NODE_IS_RE_EVENT:
                path_left.append(curr)
            curr = tree_left.parent(curr)

        path_right = []
        curr = node
        while curr != tskit.NULL:
            if ts.node(curr).flags != msprime.NODE_IS_RE_EVENT:
                path_right.append(curr)
            curr = tree_right.parent(curr)

        path_right_set = set(path_right)
        mrca = None
        for n in path_left:
            if n in path_right_set and n != node:
                mrca = n
                break

        if mrca is None:
            event['coal_left'] = None
            event['coal_right'] = None
            continue

        idx_left = path_left.index(mrca)
        idx_right = path_right.index(mrca)

        event['coal_left'] = idx_left - 1
        event['coal_right'] = idx_right - 1

    return rec_events


def events_to_dataframe(rec_events):
    sorted_events = sorted(rec_events, key=lambda e: e['time'])
    rows = []
    for i, e in enumerate(sorted_events):
        rows.append({
            'rec_number': i + 1,
            'child': e['child'],
            'descendant_samples': e['descendant_samples'],
            'descendant_nodes': e['descendant_nodes'],
            'breakpoint': e['breakpoint'],
            'time': e['time'],
            'tree_index_left': e['tree_index_left'],
            'tree_index_right': e['tree_index_right'],
            'coal_left': e['coal_left'],
            'coal_right': e['coal_right'],
        })
    return pd.DataFrame(rows)


def recombination_summary(ts):
    events = find_recombination_events(ts)
    get_descendant_samples(ts, events)
    get_tree_indices(ts, events)
    count_coalescents(ts, events)
    return events_to_dataframe(events)


def get_incomp_site_position(mts, incomp_matrix, rec_df):
    """For each recombination event, find between which pair of incompatible
    sites the breakpoint falls."""
    incompatible_sites = np.where(incomp_matrix.any(axis=1))[0]
    incomp_positions = np.array([mts.site(int(s)).position for s in incompatible_sites])

    split_points = []
    sites_left = []
    sites_right = []

    for _, row in rec_df.iterrows():
        bp = row['breakpoint']
        idx = np.searchsorted(incomp_positions, bp)
        split_points.append(idx)
        sites_left.append(incompatible_sites[idx - 1] if idx > 0 else None)
        sites_right.append(incompatible_sites[idx] if idx < len(incompatible_sites) else None)

    rec_df = rec_df.copy()
    rec_df['incomp_split_point'] = split_points
    rec_df['incomp_site_left'] = sites_left
    rec_df['incomp_site_right'] = sites_right
    return rec_df
