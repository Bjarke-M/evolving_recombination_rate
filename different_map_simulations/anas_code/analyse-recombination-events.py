import stdpopsim
import msprime
import demes
import numpy as np
import tskit
import tszip
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # For placing the colorbar
import utils

# This takes a simulated ts and gets the recombination events
# Makes a plot of events (position vs time) colouring them by proportion of YRI descendant samples
# Also plots recombination maps in the given region

# INPUTS
MAP1_CHROM = "chr8"
MAP2_CHROM = "chr12"
SEQ_START = 75_100_000
SEQ_END = 76_100_000
ZOOM_WIDTH_MB = 0.5
IN_DIRECTORY = "output"
IN_FILE = "ts_sim.trees.tsz"
OUT_FILE = "recombination_events"
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

print("\n Analyzing recombination events...")

# Create a mapping from population ID -> population name
pop_id_to_name = {
    pop.id: pop.metadata["name"]
    for pop_id, pop in enumerate(ts_mutated.populations())
}

# Create a mapping from each sample node ID -> population ID
sample_id_to_pop_id = {
    sample_id: ts_mutated.node(sample_id).population
    for sample_id in ts_mutated.samples()
}

recombination_data = []

# Find recombination events in the unsimplified ts
print("Building edge index by parent for faster lookup...")
edges_by_parent = {}
for edge in ts_mutated.edges():
    if edge.parent not in edges_by_parent:
        edges_by_parent[edge.parent] = []
    edges_by_parent[edge.parent].append(edge)

print("Iterating through nodes to find recombination events...")
node_iterator = tqdm(range(ts_mutated.num_nodes - 1), desc="Finding recombinations")
for node_id in node_iterator:
    node1 = ts_mutated.node(node_id)
    if node1.flags == msprime.NODE_IS_RE_EVENT:
        node2 = ts_mutated.node(node_id + 1)
        if node2.flags == msprime.NODE_IS_RE_EVENT and node1.time == node2.time:
            recomb_time = node1.time
            edges1 = edges_by_parent.get(node_id, [])
            edges2 = edges_by_parent.get(node_id + 1, [])

            if len(edges1) == 1 and len(edges2) == 1 and edges1[0].child == edges2[0].child:
                edge_pair = sorted(edges1 + edges2, key=lambda e: e.left)
                position = edge_pair[0].right  # breakpoint
                recomb_lineage_node = edge_pair[0].child  # RE is above this node
                tree = ts_mutated.at(position - 1)  # TODO I don't think this is totally right for reconstructed trees
                # What if there are multiple recombination and an event underneath?
                descendant_samples_left = list(tree.samples(recomb_lineage_node))
                tree = ts_mutated.at(position + 1)
                descendant_samples_right = list(tree.samples(recomb_lineage_node))
                # TODO This is definitely not right as left and right descendant samples often are different
                # if descendant_samples_left != descendant_samples_right:
                #     print("Different descendant samples to the left and right of breakpoint.")
                #     print("Will use left descendants for now.")

                if not descendant_samples_left or not descendant_samples_right:
                    continue

                counts = {"YRI": 0, "CEU": 0, "CHB": 0}
                for sample_id in descendant_samples_left:
                    pop_id = sample_id_to_pop_id.get(sample_id)
                    pop_name = pop_id_to_name.get(pop_id)
                    if pop_name:
                        counts[pop_name] += 1

                total_descendants = len(descendant_samples_left)
                proportions = {
                    name: count / total_descendants for name, count in counts.items()
                }

                recombination_data.append({
                    "position": position,
                    "time_ago": recomb_time,
                    "clade_size": total_descendants,
                    "proportions": proportions,
                    "recomb_lineage_node": recomb_lineage_node,
                    "descendant_samples_left": descendant_samples_left,
                    "descendant_samples_right": descendant_samples_right,
                })
                node_iterator.update(1)

print(f"\nFound {len(recombination_data)} recombination events in the history of the samples.")

print("  - Simplifying tree sequence...")
# tskit does not support simplifying a tree sequence
# with migrations. We can get around this by creating a new tree sequence
# from the tables with the migration table cleared.
print("  - Removing migrations to allow for simplification...")
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


print("  - Classifying recombination events...")
for event in tqdm(recombination_data, desc="Classifying events"):
    pos = event["position"]
    samples_full = event["descendant_samples_left"]
    event["type"] = -1  # Default for unclassified/error
    if not samples_full:
        continue

    samples_simple = [sample_id_map[s] for s in samples_full]  # samples in the simplified ts

    tree_before = ts_simplified.at(pos - 1)
    tree_after = ts_simplified.at(pos + 1)

    # Find MRCA of the descendant samples in each tree
    if len(samples_simple) > 1:
        mrca_before = tree_before.mrca(*samples_simple)
        mrca_after = tree_after.mrca(*samples_simple)
    else:
        mrca_before = samples_simple[0]
        mrca_after = samples_simple[0]

    if mrca_before == tskit.NULL or mrca_after == tskit.NULL:
        continue  # MRCA not found for some reason

    path_sum = utils.find_path_sum_to_common_ancestor(tree_before, mrca_before, tree_after, mrca_after)

    if path_sum == -1:
        event["type"] = 0
    elif path_sum == 0:
        event["type"] = 1
    elif path_sum == 1:
        event["type"] = 2
    elif path_sum == 2:
        event["type"] = 3
    else:
        event["type"] = 4

# Sort events by position
recombination_data.sort(key=lambda x: x['position'])


csv_filename = IN_DIRECTORY + "/" + OUT_FILE + ".csv"
print(f"\n Saving recombination data to {csv_filename}...")
with open(csv_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Position_Mb", "Time_Generations", "Clade_Size", "Prop_YRI", "Prop_CEU", "Prop_CHB", "Type", "Samples_left", "Samples_right"])
    for event in recombination_data:
        pos_mb = event['position'] / 1_000_000
        props = event['proportions']
        writer.writerow([pos_mb, event['time_ago'], event['clade_size'], props.get('YRI', 0), props.get('CEU', 0),
                         props.get('CHB', 0), event.get('type', 'N/A'),
                         ";".join([str(d) for d in event['descendant_samples_left']]),
                         ";".join([str(d) for d in event['descendant_samples_right']]),
        ])
print("Done.")

print("\n Generating plot of recombination events...")

if recombination_data:
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True, gridspec_kw={'height_ratios': [4, 1, 1]})

    type0_events = [e for e in recombination_data if e.get("type") == 0]
    type1_events = [e for e in recombination_data if e.get("type") == 1]
    type2_events = [e for e in recombination_data if e.get("type") == 2]
    type3_events = [e for e in recombination_data if e.get("type") == 3]
    type4_events = [e for e in recombination_data if e.get("type") == 4]
    print(len(type0_events), len(type1_events), len(type2_events), len(type3_events), len(type4_events))

    plot_center_mb = (SEQ_START + seq_length / 2) / 1_000_000
    xlim_min, xlim_max = (plot_center_mb - ZOOM_WIDTH_MB / 2, plot_center_mb + ZOOM_WIDTH_MB / 2)

    # Panel 1: Recombination events scatter plot
    scatter_main = None

    # Plot Type 3 and 4 events
    if type3_events:
        pos_type3 = [e['position'] / 1_000_000 for e in type3_events]
        time_type3 = [e['time_ago'] for e in type3_events]
        yri_prop_type3 = [e['proportions'].get('YRI', 0) for e in type3_events]
        scatter_main = axs[0].scatter(pos_type3, time_type3, c=yri_prop_type3, cmap='Reds', alpha=0.8, s=40, vmin=0,
                                      vmax=1, edgecolor='black', linewidths=0.5, label="Type 3 (L+R > 1)")
    if type4_events:
        pos_type4 = [e['position'] / 1_000_000 for e in type4_events]
        time_type4 = [e['time_ago'] for e in type4_events]
        yri_prop_type4 = [e['proportions'].get('YRI', 0) for e in type4_events]
        scatter_main = axs[0].scatter(pos_type4, time_type4, c=yri_prop_type4, cmap='Reds', alpha=0.8, s=40, vmin=0,
                                      vmax=1, edgecolor='black', linewidths=0.5)

    # Plot Type 2 events
    if type2_events:
        pos_type2 = [e['position'] / 1_000_000 for e in type2_events]
        time_type2 = [e['time_ago'] for e in type2_events]
        yri_prop_type2 = [e['proportions'].get('YRI', 0) for e in type2_events]
        axs[0].scatter(pos_type2, time_type2, c=yri_prop_type2, cmap='Reds', alpha=0.8, s=25, vmin=0, vmax=1,
                       edgecolor='black', linewidths=0.5, label="Type 2 (L+R = 1)", marker='^')

    # Plot Type 1 events
    if type1_events:
        pos_type1 = [e['position'] / 1_000_000 for e in type1_events]
        time_type1 = [e['time_ago'] for e in type1_events]
        yri_prop_type1 = [e['proportions'].get('YRI', 0) for e in type1_events]
        axs[0].scatter(pos_type1, time_type1, c=yri_prop_type1, cmap='Reds', alpha=0.8, s=10, vmin=0, vmax=1,
                       edgecolor='black', linewidths=0.5, label="Type 1 (L+R = 0)")

    # Plot Type 0 events
    if type0_events:
        pos_type0 = [e['position'] / 1_000_000 for e in type0_events]
        time_type0 = [e['time_ago'] for e in type0_events]
        yri_prop_type0 = [e['proportions'].get('YRI', 0) for e in type0_events]
        axs[0].scatter(pos_type0, time_type0, c=yri_prop_type0, cmap='Reds', alpha=0.8, s=10, vmin=0, vmax=1,
                       edgecolor='black', linewidths=0.5, label="Type 1 (L+R = 0)", marker='s')

    # Adjust y-axis based on all visible data points
    all_pos = [e['position'] / 1_000_000 for e in recombination_data]
    all_times = [e['time_ago'] for e in recombination_data]
    visible_times = [t for p, t in zip(all_pos, all_times) if xlim_min <= p <= xlim_max]
    if visible_times:
        max_time = max(visible_times)
        axs[0].set_ylim(bottom=0, top=max_time * 1.05)

    axs[0].axhline(y=t_iso_gen, color='dimgrey', linestyle='--', label=f"Isolation time ({ISOLATION_TIME_YEARS:,} ya)")
    axs[0].legend()

    if scatter_main is not None:
        inset_ax = inset_axes(axs[0], width="30%", height="5%", loc='upper left', bbox_to_anchor=(0.02, -0.02, 1, 1),
                              bbox_transform=axs[0].transAxes, borderpad=0)
        cbar = fig.colorbar(scatter_main, cax=inset_ax, orientation="horizontal")
        cbar.set_label("Prop. YRI samples", size=9)
        cbar.ax.tick_params(labelsize=7)

    axs[0].set_ylabel("Time (generations ago)")
    axs[0].set_title("Distribution and ancestry of recombination events")
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # Panels 2 & 3: Recombination rates
    map2_bin_mids, map2_binned_rates = utils.bin_recombination_rate(rec_map_map2_sliced, SEQ_START, SEQ_END)
    map1_bin_mids, map1_binned_rates = utils.bin_recombination_rate(rec_map_map1_sliced, SEQ_START, SEQ_END)

    visible_indices_map2 = (map2_bin_mids >= xlim_min) & (map2_bin_mids <= xlim_max)
    visible_indices_map1 = (map1_bin_mids >= xlim_min) & (map1_bin_mids <= xlim_max)
    local_max_rate = 0
    if np.any(visible_indices_map2):
        local_max_rate = max(local_max_rate, np.max(map2_binned_rates[visible_indices_map2]))
    if np.any(visible_indices_map1):
        local_max_rate = max(local_max_rate, np.max(map1_binned_rates[visible_indices_map1]))
    y_lim_top = local_max_rate * 1.15 if local_max_rate > 0 else 1

    axs[1].plot(map1_bin_mids, map1_binned_rates, color='black', drawstyle='steps-mid')
    axs[1].set_title(f"{MAP1_CHROM} recombination rate (CEU/CHB/Ancestral)")
    axs[1].set_ylabel(f"cM/Mb")
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].set_ylim(bottom=0, top=y_lim_top)

    axs[2].plot(map2_bin_mids, map2_binned_rates, color='black', drawstyle='steps-mid')
    axs[2].set_title(f"{MAP2_CHROM} recombination rate (YRI)")
    axs[2].set_ylabel(f"cM/Mb")
    axs[2].grid(True, linestyle='--', alpha=0.6)
    axs[2].set_ylim(bottom=0, top=y_lim_top)

    axs[2].set_xlabel("Genomic position (Mb)")
    axs[2].set_xlim(xlim_min, xlim_max)
    fig.tight_layout(h_pad=0.5)
    plt.savefig(IN_DIRECTORY + "/" + OUT_FILE + ".pdf")
    print(f"Plot saved to {IN_DIRECTORY}/{OUT_FILE}.pdf")
    plt.show()
else:
    print("No recombination events found to plot.")

