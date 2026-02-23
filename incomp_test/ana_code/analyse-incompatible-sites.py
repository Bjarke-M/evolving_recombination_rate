import stdpopsim
import demes
import numpy as np
import tskit
import tszip
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import utils

# This script takes an input ts and maps mutations onto 20kb regions (moving windows, centred
# every 5kb). It then finds incompatible mutation pairs within these mapped mutations.

# INPUTS
MAP1_CHROM = "chr8"
MAP2_CHROM = "chr12"
SEQ_START = 75_100_000
SEQ_END = 76_100_000
WIDTH = 20000
STEP = 5000
ZOOM_WIDTH_MB = 0.5
IN_DIRECTORY = "output"
IN_FILE = "ts_sim.trees.tsz"
OUT_FILE = IN_DIRECTORY + "/" + "genotypes"
ts_mutated = tszip.decompress(IN_DIRECTORY + "/" + IN_FILE)
ISOLATION_TIME_YEARS = 60000
MODEL = "OutOfAfrica_3G09"

seq_length = SEQ_END - SEQ_START
species = stdpopsim.get_species("HomSap")
full_model = species.get_demographic_model(MODEL)
t_iso_gen = ISOLATION_TIME_YEARS / full_model.generation_time
map2_contig_full = species.get_contig(MAP2_CHROM, genetic_map="HapMapII_GRCh37")
map1_contig_full = species.get_contig(MAP1_CHROM, genetic_map="HapMapII_GRCh37")
rec_map_map2_full = map2_contig_full.recombination_map
rec_map_map1_full = map1_contig_full.recombination_map
rec_map_map2_sliced = rec_map_map2_full.slice(SEQ_START, SEQ_END)
rec_map_map1_sliced = rec_map_map1_full.slice(SEQ_START, SEQ_END)

ts = tszip.decompress(IN_DIRECTORY + "/" + IN_FILE)
tables = ts_mutated.tables
tables.migrations.clear()
ts_mutated_no_migrations = tables.tree_sequence()
# Get node map during simplification
ts_simplified, node_map = ts_mutated_no_migrations.simplify(
    keep_unary=False, map_nodes=True
)

# Create a map from original sample IDs to simplified sample IDs
original_samples = ts_mutated_no_migrations.samples()
sample_id_map = {original_id: new_id for new_id, original_id in enumerate(original_samples)}

# Map all mutations to their sample sets (carrier sets)
all_mappable_mutations, breakpoints = utils.map_mutations_to_trees(
    ts_simplified, start_pos=SEQ_START, end_pos=SEQ_END
)

# Map mutations to genomic windows
# We pass [WIDTH] because the function expects a list of widths
windowed_positions, windowed_snp_count, windowed_snps = utils.map_mutations_to_windows(
    all_mappable_mutations,
    breakpoints,
    [WIDTH],
    STEP,
    SEQ_START,
    SEQ_END,
)

if len(windowed_positions[0]) == 0:
    print(f"No windows were generated for interval {SEQ_START}-{SEQ_END} "
          f"with width {WIDTH} and step {STEP}.")
    sys.exit()

print(f"Writing genotype matrices for {len(windowed_positions[0])} windows...")

# Get full incompatible sites matrix
incompatibility_matrix = utils.compute_incompatibility_matrix(ts_simplified)
incompatibility_data = []

# Iterate over each window, get genotypes, and write to file
for window_idx, site_ids in tqdm(windowed_snps[0].items(), desc="Writing genotype files"):

    if len(site_ids) == 0:
        # Skip empty windows
        continue

    # Get window midpoint position for filename
    window_mid_pos = int(windowed_positions[0][window_idx])
    out_file = (
        f"{OUT_FILE}_W{WIDTH}_S{int(STEP)}_"
        f"win{window_idx}_pos{window_mid_pos}.txt"
    )

    # Get the m x n genotype matrix for this window's mutations
    matrix, site_positions = utils.get_genotype_matrix(ts_simplified, site_ids)
    if matrix.size == 0:
        continue
    site_ids_ = np.fromiter(site_ids, int, len(site_ids))
    sub_matrix = incompatibility_matrix[np.ix_(site_ids_, site_ids_)]
    incompatible_count = int(np.sum(np.triu(sub_matrix, k=1)))
    incompatibility_data.append({"midpoint": window_mid_pos, "count": incompatible_count})

    # Save in fasta-like format for KwARG
    utils.save_genotypes_as_fasta_like(out_file, matrix, site_positions, window_mid_pos - WIDTH/2, window_mid_pos + WIDTH/2)

print(f"Analyzed {len(incompatibility_data)} breakpoints for incompatible mutations.")

if incompatibility_data:
    fig, ax = plt.subplots(1, 1, figsize=(8, 2), sharex=True)

    plot_center_mb = (SEQ_START + seq_length / 2) / 1_000_000
    xlim_min, xlim_max = (plot_center_mb - ZOOM_WIDTH_MB / 2, plot_center_mb + ZOOM_WIDTH_MB / 2)

    # # Panels 1 & 2: Recombination rates
    # map2_bin_mids, map2_binned_rates = utils.bin_recombination_rate(rec_map_map2_sliced, SEQ_START, SEQ_END)
    # map1_bin_mids, map1_binned_rates = utils.bin_recombination_rate(rec_map_map1_sliced, SEQ_START, SEQ_END)
    #
    # visible_indices_map2 = (map2_bin_mids >= xlim_min) & (map2_bin_mids <= xlim_max)
    # visible_indices_map1 = (map1_bin_mids >= xlim_min) & (map1_bin_mids <= xlim_max)
    # local_max_rate = 0
    # if np.any(visible_indices_map2):
    #     local_max_rate = max(local_max_rate, np.max(map2_binned_rates[visible_indices_map2]))
    # if np.any(visible_indices_map1):
    #     local_max_rate = max(local_max_rate, np.max(map1_binned_rates[visible_indices_map1]))
    # y_lim_top = local_max_rate * 1.15 if local_max_rate > 0 else 1
    #
    # axs[0].plot(map1_bin_mids, map1_binned_rates, color='black', drawstyle='steps-mid')
    # axs[0].set_ylabel(f"{MAP1_CHROM} Rec. Rate\n(cM/Mb) (CEU/CHB/ANC)")
    # axs[0].grid(True, linestyle='--', alpha=0.6)
    # axs[0].set_ylim(bottom=0, top=y_lim_top)
    #
    # axs[1].plot(map2_bin_mids, map2_binned_rates, color='black', drawstyle='steps-mid')
    # axs[1].set_ylabel(f"{MAP2_CHROM} Rec. Rate\n(cM/Mb) (YRI)")
    # axs[1].grid(True, linestyle='--', alpha=0.6)
    # axs[1].set_ylim(bottom=0, top=y_lim_top)

    breakpoints_mb = [d['midpoint'] / 1_000_000 for d in incompatibility_data]
    incompatible_counts = [d['count'] for d in incompatibility_data]
    ax.scatter(breakpoints_mb, incompatible_counts, color='black', s=20, alpha=0.7)
    ax.plot(breakpoints_mb, incompatible_counts, color='black', alpha=0.7)
    ax.set_title("Number of incompatible site pairs per window")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(bottom=0)

    ax.set_xlabel("Genomic position (Mb)")
    ax.set_xlim(xlim_min, xlim_max)
    fig.tight_layout(h_pad=0.5)
    plt.savefig(OUT_FILE + ".pdf")
    print(f"Plot saved to {OUT_FILE}.pdf")
    plt.show()
