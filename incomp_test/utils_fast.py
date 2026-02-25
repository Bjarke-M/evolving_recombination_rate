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
    Vectorized pair genotype computation.
    """
    # Find indices of the upper triangle where pairs are incompatible
    i_idx, j_idx = np.where(np.triu(pair_matrix, k=1))
    
    if len(i_idx) == 0:
        return np.empty((0, incomp_genotypes.shape[0]), dtype=int), np.empty((0, 2), dtype=int)
    
    # Extract all i and j columns at once, transpose to match shape (n_pairs, n_individuals)
    g_i = incomp_genotypes[:, i_idx].T
    g_j = incomp_genotypes[:, j_idx].T
    
    results = g_i | (g_j << 1)
    pairs = np.column_stack((i_idx, j_idx))
    
    return results, pairs


def cross_point_genotypes(all_genotypes, pairs, point):
    """
    Vectorized filtering using NumPy array masking instead of list comprehensions.
    """
    if len(pairs) == 0:
        return np.empty((0, all_genotypes.shape[1]), dtype=int)
        
    mask = (pairs[:, 0] < point) & (pairs[:, 1] >= point)
    return all_genotypes[mask]


def score_individuals(cross_genotypes, w=1):
    """
    Vectorized scoring. Eliminates the slow row-by-row np.unique calls 
    by counting the known discrete genotypes (0, 1, 2, 3) across the whole matrix at once.
    """
    n_pairs, n_ind = cross_genotypes.shape
    scores = np.zeros(n_ind, dtype=float)
    
    if n_pairs == 0:
        return scores

    # Base score: genotype == 3
    scores += (cross_genotypes == 3).sum(axis=0)

    # Unique gamete bonus
    for v in range(4):
        is_v = (cross_genotypes == v)           # Boolean mask of where value 'v' is
        v_counts = is_v.sum(axis=1)             # How many times 'v' appears in each pair
        unique_v_mask = (v_counts == 1)         # Mask of pairs where 'v' only appears once
        
        # Add 'w' to the individuals holding the unique 'v' in those specific pairs
        scores += (is_v & unique_v_mask[:, None]).sum(axis=0) * w

    return scores


def compute_score_matrix(incomp_genotypes, sub_incomp):
    n_nodes, n_sites = incomp_genotypes.shape
    all_genos, pairs = incomp_pair_genotypes(incomp_genotypes, sub_incomp)

    score_mat = np.zeros((n_sites - 1, n_nodes))
    for pt in range(1, n_sites):
        cg = cross_point_genotypes(all_genos, pairs, pt)
        if cg.shape[0] > 0:
            score_mat[pt - 1] = score_individuals(cg, w=1.0)

    return score_mat


def score_at_split_points(genotype_matrix, incomp_matrix, w=1):
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

        # The huge speedup will be felt here
        result = score_at_split_points(genotype_matrix, current_matrix, w=w)

        iterations.append({
            'incomp_sites': result['incomp_sites'],
            'scores': result['scores'],
            'total': result['scores'].sum(axis=0),
            'removed_site': int(youngest_site),
            'removed_time': float(youngest_time),
            'n_incomp': len(incomp_sites),
        })

        # Zero out removed site
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
