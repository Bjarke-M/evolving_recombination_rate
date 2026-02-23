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


def add_mutations(ts):
    tables = ts.dump_tables()
    tables.mutations.clear()
    tables.sites.clear()
    s = 0
    for t in ts.trees():
        left, right = t.interval
        coal_nodes = [n for n in t.nodes() if t.num_children(n) >= 2]
        n_coal = len(coal_nodes)
        for k, node in enumerate(coal_nodes):
            pos = left + (k + 1) * (right - left) / (n_coal + 1)
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


def compute_incompatibility_matrix_batched(ts, recsites=set(), batch_size=1000):
    num_sites = ts.num_sites

    # 1. Filter and Prepare Genotypes
    # Identify which sites we actually need to calculate
    valid_mask = np.ones(num_sites, dtype=bool)
    if recsites:
        valid_mask[list(recsites)] = False

    valid_indices = np.where(valid_mask)[0]

    # Get genotypes only for valid sites
    G_valid = ts.genotype_matrix()[valid_mask].astype(np.float32)

    # Pre-calculate the total mutation count for every valid site
    # This is needed for the logic check later
    count_1_global = G_valid.sum(axis=1)

    # Initialize the result matrix
    matrix = np.zeros((num_sites, num_sites), dtype=np.int8)

    # 2. Iterate in Batches
    num_valid = len(valid_indices)

    # We loop through the valid sites in chunks (e.g., 1000 rows at a time)
    for start in tqdm(range(0, num_valid, batch_size), desc="Computing Matrix"):
        end = min(start + batch_size, num_valid)

        # Get the batch of rows (subset of sites)
        G_batch = G_valid[start:end]

        # Matrix Multiply: Batch vs All Valid Sites
        # This gives intersections for the current batch against everyone
        count_11_batch = G_batch @ G_valid.T

        # Get the "total 1s" for just this batch
        count_1_batch = count_1_global[start:end]

        # 3. Apply the Four-Gamete Logic (Vectorized)
        # Condition A: Has (1,1) -> Intersection > 0
        has_11 = count_11_batch > 0

        # Condition B: Has (1,0) -> Batch site has more 1s than the intersection
        # broadcast (batch_size, 1) against (batch_size, num_valid)
        has_10 = count_1_batch[:, np.newaxis] > count_11_batch

        # Condition C: Has (0,1) -> Target site has more 1s than the intersection
        # broadcast (1, num_valid) against (batch_size, num_valid)
        has_01 = count_1_global[np.newaxis, :] > count_11_batch

        # Combine logic
        incompatible_batch = (has_11 & has_10 & has_01)

        # 4. Map back to the full N x N matrix
        # We need to map the batch rows (relative) to the real rows (absolute)
        batch_real_indices = valid_indices[start:end]

        # Use np.ix_ to map the rectangular batch into the full square matrix
        matrix[np.ix_(batch_real_indices, valid_indices)] = incompatible_batch.astype(np.int8)

    # 5. Cleanup
    np.fill_diagonal(matrix, 0)

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


# --- Alternative scoring functions ---

def score_inverse_frequency(cross_genotypes):
    """Score each individual by the inverse frequency of their gamete
    at each incompatible pair.
    """
    n_pairs, n_ind = cross_genotypes.shape
    scores = np.zeros(n_ind)

    for row_idx in range(n_pairs):
        row = cross_genotypes[row_idx, :]
        values, counts = np.unique(row, return_counts=True)
        freq_map = dict(zip(values, counts))
        for ind in range(n_ind):
            scores[ind] += 1.0 / freq_map[row[ind]]

    return scores


def score_log_frequency(cross_genotypes):
    """Score each individual by the negative log frequency of their gamete
    at each incompatible pair.
    """
    n_pairs, n_ind = cross_genotypes.shape
    scores = np.zeros(n_ind)

    for row_idx in range(n_pairs):
        row = cross_genotypes[row_idx, :]
        values, counts = np.unique(row, return_counts=True)
        freq_map = dict(zip(values, counts))
        for ind in range(n_ind):
            scores[ind] += -np.log(freq_map[row[ind]] / n_ind)

    return scores


def tag_rarest_gamete(cross_genotypes):
    """For each incompatible pair, find the rarest gamete and tag the
    individuals that carry it.
    """
    n_pairs, n_ind = cross_genotypes.shape
    tagged = np.zeros((n_pairs, n_ind), dtype=bool)

    for row_idx in range(n_pairs):
        row = cross_genotypes[row_idx, :]
        values, counts = np.unique(row, return_counts=True)
        rarest_gamete = values[np.argmin(counts)]
        tagged[row_idx, :] = (row == rarest_gamete)

    vote_counts = tagged.sum(axis=0)
    return tagged, vote_counts


def score_rarest_gamete(cross_genotypes, w=0):
    """Score each individual based on rarest-gamete tagging.

    For each incompatible pair:
      - +1 if the individual carries gamete 11 (same as original)
      - +w if the individual carries the rarest gamete
    """
    n_pairs, n_ind = cross_genotypes.shape
    scores = np.zeros(n_ind)

    scores += (cross_genotypes == 3).sum(axis=0).astype(float)

    for row_idx in range(n_pairs):
        row = cross_genotypes[row_idx, :]
        values, counts = np.unique(row, return_counts=True)
        rarest_gamete = values[np.argmin(counts)]
        is_rarest = (row == rarest_gamete)
        scores[is_rarest] += w

    return scores


def score_combined(cross_genotypes, w_rarest=1, w_eleven=1, w_invfreq=1):
    """Combined scoring: rarest gamete + gamete 11 + inverse frequency.

    For each incompatible pair, each individual gets:
      - +w_eleven  if their gamete is 11
      - +w_rarest  if they carry the rarest gamete
      - +w_invfreq * (1/count) where count is how many individuals share their gamete

    Parameters
    ----------
    cross_genotypes : array of shape (n_pairs, n_individuals), values 0-3
    w_rarest : float, weight for carrying the rarest gamete
    w_eleven : float, weight for carrying gamete 11
    w_invfreq : float, weight for the inverse-frequency component

    Returns
    -------
    scores : array of shape (n_individuals,)
    """
    n_pairs, n_ind = cross_genotypes.shape
    scores = np.zeros(n_ind)

    # +w_eleven for gamete 11
    scores += w_eleven * (cross_genotypes == 3).sum(axis=0).astype(float)

    for row_idx in range(n_pairs):
        row = cross_genotypes[row_idx, :]
        values, counts = np.unique(row, return_counts=True)
        freq_map = dict(zip(values, counts))

        # Rarest gamete
        rarest_gamete = values[np.argmin(counts)]

        for ind in range(n_ind):
            # +w_rarest for rarest gamete
            if row[ind] == rarest_gamete:
                scores[ind] += w_rarest
            # +w_invfreq * 1/count
            scores[ind] += w_invfreq / freq_map[row[ind]]

    return scores
