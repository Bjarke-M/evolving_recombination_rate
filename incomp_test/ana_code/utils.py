import tszip
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import stdpopsim
import tskit

def map_mutations_to_trees(ts, min_carriers=3, map_threshold=1000000, start_pos=None, end_pos=None):
    """
    Maps all mutations in a tree sequence to the trees where their topology exists.

    This works in two stages:
    1.  Build a dictionary mapping mutation carrier sets (frozensets of sample nodes)
        to the mutation IDs that produced them. This iterates over the entire tree sequence.
    2.  Iterate over trees in the specified genomic region [start_pos, end_pos].
        For each tree, find all unique mutations from the map created in stage 1
        that "match" a node's topology in that tree (i.e., the node subtends
        the same set of samples as the mutation).

    :param tskit.TreeSequence ts: The input tree sequence.
    :param int min_carriers: The minimum number of samples a mutation must
        have to be considered for mapping.
    :param float start_pos: The genomic start position for mapping mutations
        to trees. If None, defaults to the start of the sequence.
    :param float end_pos: The genomic end position for mapping mutations
        to trees. If None, defaults to the end of the sequence.
    :return: A tuple (all_mappable_mutations, breakpoints) where:
        - all_mappable_mutations (dict): A dictionary mapping tree index (int)
          to a sorted list of unique mutation IDs (int) mappable to that tree.
        - breakpoints (np.ndarray): The array of tree sequence breakpoints.
    """
    
    print("Building map of mutations by carrier set...")
    mutations_by_carrier_set = defaultdict(list)
    with tqdm(total=ts.num_trees) as pbar:
        for tree in ts.trees():
            for mut in tree.mutations():
                carrier_samples = frozenset(tree.samples(mut.node))
                if carrier_samples and min_carriers <= len(carrier_samples) <= ts.num_samples:
                    mutations_by_carrier_set[carrier_samples].append(mut.site)
            pbar.update(1)
    print(f"Found {len(mutations_by_carrier_set)} unique carrier sets.")

    print(f"Mapping mutations to trees in interval [{start_pos}, {end_pos}]...")
    all_mappable_mutations = {}
    
    if start_pos is None:
        start_pos = ts.first().interval[0]
    if end_pos is None:
        end_pos = ts.sequence_length

    breakpoints = ts.breakpoints(as_array=True)
    
    with tqdm(total=ts.num_trees) as pbar:
        for tree in ts.trees():
            # Only process trees that (partially) overlap the interval
            if tree.interval[1] >= start_pos and tree.interval[0] <= end_pos:
                found_muts_for_this_tree = set()
                for node in tree.nodes():
                    if not tree.is_sample(node):
                        clade_samples = frozenset(tree.samples(node))
                        if clade_samples in mutations_by_carrier_set:
                            found_muts_for_this_tree.update([m for m in mutations_by_carrier_set[clade_samples] if tree.interval[0] - map_threshold <= ts.site(m).position <= tree.interval[1] + map_threshold])
                all_mappable_mutations[tree.index] = sorted(list(found_muts_for_this_tree))
            pbar.update(1)
            
    if not all_mappable_mutations:
        raise ValueError(
            f"No trees were found and mapped in the interval [{start_pos}, {end_pos}]. ")

    print("Mapping done.")
    return all_mappable_mutations, breakpoints


def get_staggered_intervals(start, width, step):
    """
    Generator function to yield staggered window intervals.

    :param float start: The starting position of the first interval.
    :param float width: The width of each interval.
    :param float step: The step size to move the window.
    :yield: tuple (current_start, current_start + width)
    """
    current_start = start
    while True:
        yield (current_start, current_start + width)
        current_start += step


def map_mutations_to_windows(
    all_mappable_mutations, breakpoints, widths, step, start_pos, end_pos
):
    """
    Aggregates unique mappable mutations into genomic windows.

    :param dict all_mappable_mutations: The output from map_mutations_to_trees.
    :param np.ndarray breakpoints: The array of tree sequence breakpoints.
    :param list widths: A list of window widths (int) to analyze.
    :param float step: The step size for the sliding windows.
    :param float start_pos: The genomic start position for windowing.
    :param float end_pos: The genomic end position for windowing.
    :return: A tuple (windowed_positions, windowed_snp_count) where:
        - windowed_positions (np.ndarray): 2D array storing the midpoint
          position for each window.
        - windowed_snp_count (np.ndarray): 2D array storing the count of
          unique mappable mutations in each window.
    """
    print("Mapping mutations to windows...")
    
    final_positions = []
    final_snp_count = []
    final_windowed_snp = []
    
    min_mapped_idx = min(all_mappable_mutations.keys())
    max_mapped_idx = max(all_mappable_mutations.keys())

    for w, width in enumerate(widths):
        count = 0
        total = int(np.floor((end_pos - width - start_pos) / step + 1))
        windowed_positions = np.zeros(total)
        windowed_snp_count = np.zeros(total)
        windowed_snp = {}
        
        with tqdm(total=total, desc=f"Width {width}bp") as pbar:
            for start, end in get_staggered_intervals(start_pos, width, step):
                if end > end_pos:
                    break
                    
                # Find tree indices corresponding to the window
                try:
                    i = int(np.where(breakpoints < start)[0][-1])
                    j = int(np.where(breakpoints > end)[0][0] - 1)
                except IndexError:
                    # Window is outside the range of breakpoints
                    continue

                # Skip windows that start before our mapped region
                if i < min_mapped_idx:
                    continue
                # Stop if windows go beyond our mapped region
                if j > max_mapped_idx:
                    break

                n = set()
                for k in range(i, j + 1):
                    # Check if tree index 'k' was mapped (it might be skipped
                    # if it was outside the [start_pos, end_pos] in the first func)
                    if k in all_mappable_mutations:
                        n.update(set(all_mappable_mutations[k]))
                        
                windowed_positions[count] = (start + end) / 2
                windowed_snp_count[count] = len(n)
                windowed_snp[count] = n
                count += 1
                pbar.update(1)
        
        final_positions.append(windowed_positions)
        final_snp_count.append(windowed_snp_count)
        final_windowed_snp.append(windowed_snp)

    return final_positions, final_snp_count, final_windowed_snp


def get_genotype_matrix(ts, site_ids):
    """
    Get the genotype matrix for a specific set of mutation IDs.
    
    This function uses the tskit.Variant class to efficiently decode
    genotypes for only the sites corresponding to the provided mutation IDs.

    :param tskit.TreeSequence ts: The input tree sequence.
    :param set mutation_ids: A set of mutation IDs.
    :return: A tuple (genotype_matrix, site_positions) where
        - genotype_matrix (np.ndarray): The m x n genotype matrix,
          where m is the number of sites and n is the number of samples.
        - site_positions (list): The list of site positions (length m),
          sorted by position.
    """
    if not site_ids:
        # Return empty matrix and list
        return np.array([]).reshape(0, ts.num_samples), []
    
    genotypes_list = []
    site_pos_list = []
    variant = tskit.Variant(ts)
    site_ids = sorted(list(site_ids))
    
    for site_id in site_ids:
        # Decode the variant information for this specific site
        variant.decode(site_id)
        # Append the genotype array
        genotypes_list.append(np.array(variant.genotypes, copy=True))
        # Append the site position
        site_pos_list.append(int(variant.site.position))
    
    # Stack the individual genotype arrays into a single matrix
    if not genotypes_list:
         return np.array([]).reshape(0, ts.num_samples), [] # [cite: 51]

    genotype_matrix = np.vstack(genotypes_list)
    
    return genotype_matrix, site_pos_list


def save_genotypes_as_fasta_like(filepath, genotype_matrix, site_positions, start, end, sample_prefix="Seq"):
    """
    Saves a genotype matrix in a FASTA-like format, as requested.

    :param str filepath: The path to the output file.
    :param np.ndarray genotype_matrix: The m x n (sites x samples) genotype matrix.
    :param list site_positions: A list of m site positions.
    :param str sample_prefix: The prefix for each sample sequence (e.g., "Seq").
    """
    
    # Create a mapping from integer genotypes to string representation
    # [cite_start]tskit.MISSING_DATA is -1 [cite: 398]
    genotype_map = {0: '0', 1: '1', -1: 'x'}
    
    with open(filepath, 'w') as f:
        # Iterate over each sample's genotypes (each row in the transposed matrix)
        for sample_idx, genotype_row in enumerate(np.transpose(genotype_matrix)):
            # Write the header
            # print(len(genotype_row), len(site_positions))
            f.write(f"#> {sample_prefix}{sample_idx + 1}\n")
            
            # Convert the row of integers to a string of '0', '1', 'x'
            # Use .get() for safety, defaulting to '?' if an unexpected value appears
            genotype_string = "".join(
                [genotype_map.get(g, '?') for g in genotype_row]
            )
            
            # Write the genotype string
            f.write(f"{genotype_string}\n")
        
        # Write the final positions line
        positions_str = ' '.join(map(str, site_positions))
        f.write(f"#positions: {positions_str}\n")

        # Plot positions of the mutations mapping to this window
        fig, ax = plt.subplots( nrows=1, ncols=1 )
        ax.hist(site_positions, bins=100)
        ax.vlines(x=[start, end], ymin=0, ymax=10, linestyle=":", color="red")
        fig.savefig(filepath + '.png', bbox_inches='tight')
        plt.close(fig)

def find_path_sum_to_common_ancestor(tree_before, mrca_before, tree_after, mrca_after):
    """
    Finds the first common ancestor of two nodes from different trees and returns
    the sum of the path lengths from the original nodes to that ancestor.
    """
    # Get path to root for mrca_before in tree_before
    path_before = []
    curr = mrca_before
    while curr != tskit.NULL:
        curr = tree_before.parent(curr)
        path_before.append(curr)

    ancestors_before = set(path_before)

    # Traverse up from mrca_after in tree_after to find the first common ancestor
    R = 0
    curr = mrca_after
    while curr != tskit.NULL:
        curr = tree_after.parent(curr)
        R += 1
        if curr in ancestors_before:
            # Now find L, which is the index of the fca in the path from mrca_before
            try:
                L = 1 + path_before.index(curr)
                # if (L == 1 and R == 2) or (L == 2 and R == 1):
                #     print(mrca_before, mrca_after)
                #     print(path_before)
                #     print(curr)
                #     print(L, R)
                #     sys.exit()
                # print(L, R, L + R - 2)
                return L + R - 2
            except ValueError:
                # Should not happen if curr is in the set
                return -1

    return -1  # Should not be reached if there's a common root


def bin_recombination_rate(rec_map, seq_start, seq_end, bin_width=1000):
    bins = np.arange(seq_start, seq_end, bin_width)
    bin_mids = bins + bin_width / 2
    binned_rates = np.zeros_like(bin_mids, dtype=float)
    map_pos = rec_map.position
    map_rates = rec_map.rate
    for i, bin_start in enumerate(bins):
        bin_end = bin_start + bin_width
        j_start = np.searchsorted(map_pos, bin_start, side='right') - 1
        j_start = max(0, j_start)
        j_end = np.searchsorted(map_pos, bin_end, side='left')
        total_length = 0
        weighted_rate_sum = 0
        for j in range(j_start, min(j_end + 1, len(map_pos) - 1)):
            seg_start, seg_end, seg_rate = map_pos[j], map_pos[j + 1], map_rates[j]
            overlap_start = max(seg_start, bin_start)
            overlap_end = min(seg_end, bin_end)
            overlap_len = overlap_end - overlap_start
            if overlap_len > 0:
                weighted_rate_sum += seg_rate * overlap_len
                total_length += overlap_len
        if total_length > 0:
            binned_rates[i] = weighted_rate_sum / total_length
        else:
            idx = np.searchsorted(map_pos, bin_mids[i], side='right') - 1
            if 0 <= idx < len(map_rates):
                binned_rates[i] = map_rates[idx]
    return bin_mids / 1_000_000, binned_rates


def compute_incompatibility_matrix(ts):
    num_sites = ts.num_sites
    genotypes = ts.genotype_matrix()
    matrix = np.zeros((num_sites, num_sites), dtype=np.int8)
    for i in tqdm(range(num_sites), desc="  - Pre-computing 4GT matrix"):
        for j in range(i + 1, num_sites):
            g1 = genotypes[i, :]
            g2 = genotypes[j, :]
            gametes = g1 | (g2 << 1)
            gametes = np.append(gametes, 0)  # Known root type 00
            if len(np.unique(gametes)) == 4:
                matrix[i, j] = 1
                matrix[j, i] = 1
    return matrix


def get_mappable_sites_by_clade(tree, carriers_to_sites_map):
    mappable_sites = set()
    for node in tree.nodes():
        clade_samples = frozenset(tree.samples(node))
        if clade_samples in carriers_to_sites_map:
            for site_id in carriers_to_sites_map[clade_samples]:
                mappable_sites.add(site_id)
    if frozenset() in carriers_to_sites_map:
        for site_id in carriers_to_sites_map[frozenset()]:
            mappable_sites.add(site_id)
    return np.array(list(mappable_sites), dtype=np.int32)
