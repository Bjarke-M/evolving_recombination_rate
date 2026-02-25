import tskit
import msprime
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def count_coalescents_to_common_ancestor(ts, idx_left, idx_right, node):
    tree_left = ts.at_index(idx_left)
    tree_right = ts.at_index(idx_right)

    # ancestery path for node in left tree
    anc_path_left = []
    curr = node
    while curr != tskit.NULL:
        anc_path_left.append(curr)
        curr = tree_left.parent(curr)

    # ancestery path for node in right tree
    anc_path_right = []
    curr = node
    while curr != tskit.NULL:
        anc_path_right.append(curr)
        curr = tree_right.parent(curr)

    print(anc_path_left)
    print(anc_path_right)
    mrca = [x for x in anc_path_left if x in anc_path_right and x != node][0]
    coal_c_left = anc_path_left.index(mrca) - 2
    coal_c_right = anc_path_right.index(mrca) - 2
    print(mrca)
    print('coalescents left:', coal_c_left)
    print('coalescents right:', coal_c_right)

    return (coal_c_left, coal_c_right)


def add_mutations(ts, seed=None):
    tables = ts.dump_tables()
    tables.mutations.clear()
    tables.sites.clear()
    s = 0
    used_int_positions = set()
    if seed:
        np.random.seed(seed)
    for t in ts.trees():
        left, right = t.interval
        coal_nodes = [n for n in t.nodes() if t.num_children(n) >= 2]
        for node in coal_nodes:
            pos = np.random.uniform(left, right)
            # Skip if this integer position is already taken 
            if int(pos) in used_int_positions:
                continue
            used_int_positions.add(int(pos))
            tables.sites.add_row(position=pos, ancestral_state="A")
            tables.mutations.add_row(site=s, node=node, derived_state="G", time=None)
            s += 1

    tables.sort()
    return tables.tree_sequence()


def find_recsites(ts):
    recsites = set()
    for s in ts.sites():
        if len(s.mutations) > 1:
            recsites.add(s.id)
    return recsites


def compute_incompatibility_matrix(ts, recsites=set()):
    num_sites = ts.num_sites
    genotypes = ts.genotype_matrix()
    matrix = np.zeros((num_sites, num_sites), dtype=np.int8)
    for i in tqdm(range(num_sites), desc="Computing incompatibility"):
        if i not in recsites:
            for j in range(i + 1, num_sites):
                if j not in recsites:
                    g1 = genotypes[i, :]
                    g2 = genotypes[j, :]
                    gametes = g1 | (g2 << 1)
                    gametes = np.append(gametes, 0)
                    if len(np.unique(gametes)) == 4:
                        matrix[i, j] = 1
                        matrix[j, i] = 1
    return matrix


def compute_incompatibility_matrix_2(genotype_matrix):
    num_sites = genotype_matrix.shape[0]
    genotypes = genotype_matrix
    matrix = np.zeros((num_sites, num_sites), dtype=np.int8)
    for i in tqdm(range(num_sites), desc="Computing incompatibility"):
            for j in range(i + 1, num_sites):
                    g1 = genotypes[i, :]
                    g2 = genotypes[j, :]
                    gametes = g1 | (g2 << 1)
                    gametes = np.append(gametes, 0)
                    if len(np.unique(gametes)) == 4:
                        matrix[i, j] = 1
                        matrix[j, i] = 1
    return matrix




def incomp_pair_genotypes(incomp_genotypes, pair_matrix):
    """
    For each incompatible pair in pair_matrix, compute the genotype (0-3)
    for each individual.

    Parameters
    ----------
    incomp_genotypes : array of shape (n_individuals, n_sites), values 0 or 1
    pair_matrix : array of shape (n_sites, n_sites), 1 = incompatible pair

    Returns
    -------
    genotypes : array of shape (n_pairs, n_individuals)
    pairs : list of (i, j) tuples, one per row
    """
    n_sites = incomp_genotypes.shape[1]
    results = []
    pairs = []
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            if pair_matrix[i, j]:
                row = incomp_genotypes[:, i] | (incomp_genotypes[:, j] << 1)
                results.append(row)
                pairs.append((i, j))
    if len(results) == 0:
        return np.empty((0, incomp_genotypes.shape[0]), dtype=int), []
    return np.array(results), pairs


def cross_point_genotypes(all_genotypes, pairs, point):
    """
    Filter the full pair genotype matrix to only pairs that cross the point.

    Parameters
    ----------
    all_genotypes : array of shape (n_pairs, n_individuals), values 0-3
        Output of incomp_pair_genotypes.
    pairs : list of (i, j) tuples matching rows of all_genotypes
    point : int, the split point

    Returns
    -------
    genotypes : array of shape (n_crossing_pairs, n_individuals)
    """
    mask = [i < point and j >= point for i, j in pairs]
    if not any(mask):
        return np.empty((0, all_genotypes.shape[1]), dtype=int)
    return all_genotypes[mask]


def score_individuals(cross_genotypes, w=1):
    """
    Score each individual based on cross-point genotypes.

    For each site pair:
      - genotype == 3 (11): +1 to that individual's score
      - individual is the only one with its genotype in that pair: +w

    Parameters
    ----------
    cross_genotypes : array of shape (n_pairs, n_individuals), values 0-3
    w : float, weight for being the unique holder of a genotype

    Returns
    -------
    scores : array of shape (n_individuals,)
    """
    n_pairs, n_ind = cross_genotypes.shape
    scores = np.zeros(n_ind)

    scores += (cross_genotypes == 3).sum(axis=0)

    for row_idx in range(cross_genotypes.shape[0]):
        row_data = cross_genotypes[row_idx, :]
        values, counts = np.unique(row_data, return_counts=True)
        unique_values_in_row = values[counts == 1]
        cols = np.where(np.isin(row_data, unique_values_in_row))[0]
        for col_idx in cols:
            scores[col_idx] += w

    return scores


def compute_score_matrix(incomp_genotypes, sub_incomp):
    """
    Compute scores for all nodes at every split point.

    Parameters
    ----------
    incomp_genotypes : array (n_nodes, n_incomp_sites)
        Transposed genotype matrix for incompatible sites only.
    sub_incomp : array (n_incomp_sites, n_incomp_sites)
        Incompatibility sub-matrix for those sites.

    Returns
    -------
    score_mat : array (n_split_points, n_nodes)
        Score for each node at each split point.
    """
    n_nodes, n_sites = incomp_genotypes.shape
    all_genos, pairs = incomp_pair_genotypes(incomp_genotypes, sub_incomp)

    score_mat = np.zeros((n_sites - 1, n_nodes))
    for pt in range(1, n_sites):
        cg = cross_point_genotypes(all_genos, pairs, pt)
        if cg.shape[0] > 0:
            score_mat[pt - 1] = score_individuals(cg, w=1.0)

    return score_mat


def score_at_split_points(genotype_matrix, incomp_matrix, w=1):
    """
    Score each node at each split point between incompatible sites.

    Parameters
    ----------
    genotype_matrix : ndarray of shape (n_sites, n_nodes)
        Full genotype matrix (0/1 values).
    incomp_matrix : ndarray of shape (n_sites, n_sites)
        Full incompatibility matrix (symmetric, 0/1 values).
    w : float
        Weight for unique-gamete bonus in score_individuals.

    Returns
    -------
    dict with keys:
        'scores' : ndarray of shape (n_split_points, n_nodes)
        'incomp_sites' : ndarray of original site indices that are incompatible
    """
    incomp_sites = np.where(incomp_matrix.any(axis=1))[0]
    n_nodes = genotype_matrix.shape[1]

    if len(incomp_sites) < 2:
        return {
            'scores': np.empty((0, n_nodes)),
            'incomp_sites': incomp_sites,
        }

    incomp_geno = genotype_matrix[incomp_sites, :].T
    sub_incomp = incomp_matrix[np.ix_(incomp_sites, incomp_sites)]
    all_genos, pairs = incomp_pair_genotypes(incomp_geno, sub_incomp)

    n_incomp = len(incomp_sites)
    score_mat = np.zeros((n_incomp - 1, n_nodes))

    for pt in tqdm(range(1, n_incomp), desc="  Split points", leave=False):
        cg = cross_point_genotypes(all_genos, pairs, pt)
        if cg.shape[0] > 0:
            score_mat[pt - 1] = score_individuals(cg, w=w)

    return {
        'scores': score_mat,
        'incomp_sites': incomp_sites,
    }


def iterative_removal_scoring(genotype_matrix, incomp_matrix, site_mut_times, w=1):
    """
    Iteratively remove the youngest incompatible site, re-score, and collect
    results.

    At each iteration the function:
      1. Identifies currently incompatible sites.
      2. Scores all nodes at every split point (via score_at_split_points).
      3. Records the scores and which site was removed.
      4. Zeros out the youngest incompatible site's row and column.

    Works for both simulated trees (genotype matrix includes internal nodes)
    and Relate trees (samples only).

    Parameters
    ----------
    genotype_matrix : ndarray of shape (n_sites, n_nodes)
        Full genotype matrix.
    incomp_matrix : ndarray of shape (n_sites, n_sites)
        Full incompatibility matrix (copied internally).
    site_mut_times : ndarray of shape (n_sites,)
        Mutation time for each site. The site with the smallest time among
        currently incompatible sites is removed each iteration.
    w : float
        Weight for unique-gamete bonus in score_individuals.

    Returns
    -------
    list of dict, one per iteration. Each dict contains:
        'incomp_sites' : ndarray -- currently incompatible site indices
        'scores' : ndarray (n_split_points, n_nodes) -- per-split-point scores
        'total' : ndarray (n_nodes,) -- scores summed across split points
        'removed_site' : int -- index of the site removed this iteration
        'removed_time' : float -- mutation time of the removed site
        'n_incomp' : int -- number of incompatible sites this iteration
    """
    current_matrix = incomp_matrix.copy()
    iterations = []

    n_initial = np.count_nonzero(current_matrix.any(axis=1))
    pbar = tqdm(total=max(n_initial - 1, 0), desc="Iterative scoring")

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


# --- Recombination event summary functions ---

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
            breakpoint = min(e.right for e in edges_left)
            events.append({
                'rec_node_left': node_id,
                'rec_node_right': partner,
                'child': child,
                'breakpoint': breakpoint,
                'time': node.time,
            })

    return events


def get_descendant_samples(ts, rec_events):
    for event in rec_events:
        tree = ts.at(event['breakpoint'] - 1)
        event['descendant_samples'] = sorted(tree.samples(event['child']))
        # All descendant nodes (samples + internal), excluding RE nodes
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
    sites the breakpoint falls.
    """
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
