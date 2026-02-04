import stdpopsim
import msprime
import numpy as np
import tszip

# This simulates a ts (with record_full_arg = True) with three populations
# One of the populations is isolated until ISOLATION_TIME_YEARS ago and until that point has
# recombination map corresponding to the map for MAP2_CHROM
# The two other populations (and the merged population after ISOLATION_TIME_YEARS ago) use
# recombination map corresponding to MAP1_CHROM

# INPUTS
MAP1_CHROM = "chr8"
MAP2_CHROM = "chr12"
SEQ_START = 75_100_000
SEQ_END = 76_100_000
YRI_N_SAMPLES = 10
CEU_N_SAMPLES = 10
CHB_N_SAMPLES = 10
OUT_DIRECTORY = "output"
OUT_NAME = "ts_sim"

ISOLATION_TIME_YEARS = 60000
MODEL = "OutOfAfrica_3G09"

print(f"-> Running with stdpopsim version: {stdpopsim.__version__}")
print(f"-> Running with msprime version: {msprime.__version__}")

print("Setting up global parameters...")
species = stdpopsim.get_species("HomSap")
full_model = species.get_demographic_model(MODEL)

# Define the genomic region for the simulation
seq_length = SEQ_END - SEQ_START
print(f"Simulating genomic region: {SEQ_START / 1e6:.1f}Mb - {SEQ_END / 1e6:.1f}Mb (Length: {seq_length / 1e3:.0f}kb)")

# Get contigs and their recombination maps
map2_contig_full = species.get_contig(MAP2_CHROM, genetic_map="HapMapII_GRCh37")
map1_contig_full = species.get_contig(MAP1_CHROM, genetic_map="HapMapII_GRCh37")
rec_map_map2_full = map2_contig_full.recombination_map
rec_map_map1_full = map1_contig_full.recombination_map

# Slice the recombination maps to the desired region
rec_map_map2_sliced = rec_map_map2_full.slice(SEQ_START, SEQ_END)
rec_map_map1_sliced = rec_map_map1_full.slice(SEQ_START, SEQ_END)

# Use the mutation rate from the contig object.
MUTATION_RATE = map2_contig_full.mutation_rate

# Define timing and sample sizes (TEST CASE)
generation_time = full_model.generation_time
t_iso_gen = ISOLATION_TIME_YEARS / generation_time
yri_samples = {"YRI": YRI_N_SAMPLES}
ceu_chb_samples = {"CEU": CEU_N_SAMPLES, "CHB": CHB_N_SAMPLES}
full_graph = full_model.model.to_demes()

# Model B: The original, unmodified demography for Phase 2
msprime_demography_full = msprime.Demography.from_demes(full_graph)

# Model A: A modified demography for Phase 1 where YRI does not migrate
# We create a copy and then set the migration rates involving YRI to zero
print("  - Creating modified demography with YRI migration rate set to 0...")
msprime_demography_modified = msprime.Demography.from_demes(full_graph)
msprime_demography_modified.set_symmetric_migration_rate(populations=["YRI", "CEU"], rate=0)
msprime_demography_modified.set_symmetric_migration_rate(populations=["YRI", "CHB"], rate=0)

# Phase 1
print(f"\n PHASE 1: Simulating from present to {ISOLATION_TIME_YEARS:,} years ago...")

# Simulation A (YRI with map2 recombination map)
print(f"  - Simulating YRI with {MAP2_CHROM} recombination map (YRI isolated)...")
ts_yri = msprime.sim_ancestry(
    samples=yri_samples,
    demography=msprime_demography_modified,
    recombination_rate=rec_map_map2_sliced,
    end_time=t_iso_gen,
    record_migrations=True,
    record_full_arg=True,
    random_seed=10
)
print(f"  - YRI simulation (Phase 1) has {ts_yri.num_migrations} migrations.")

# Simulation B (CEU/CHB with map1 recombination map)
print(f"  - Simulating CEU & CHB with {MAP1_CHROM} recombination map (YRI isolated)...")
ts_ceu_chb = msprime.sim_ancestry(
    samples=ceu_chb_samples,
    demography=msprime_demography_modified,
    recombination_rate=rec_map_map1_sliced,
    end_time=t_iso_gen,
    record_migrations=True,
    record_full_arg=True,
    random_seed=20
)
print(f"  - CEU/CHB simulation (Phase 1) has {ts_ceu_chb.num_migrations} migrations.")

# Combine Phase 1
print("\n  - Combining results to create initial state for Phase 2...")
tables = ts_ceu_chb.tables
tables_other = ts_yri.tables

# Calculate offsets needed to merge tables
node_offset = len(tables.nodes)
individual_offset = len(tables.individuals)

# Append individuals
tables.individuals.append_columns(
    flags=tables_other.individuals.flags,
    location=tables_other.individuals.location,
    location_offset=tables_other.individuals.location_offset,
    parents=tables_other.individuals.parents,
    parents_offset=tables_other.individuals.parents_offset,
    metadata=tables_other.individuals.metadata,
    metadata_offset=tables_other.individuals.metadata_offset,
)

# Append nodes. Population IDs are already correct relative to each other
tables.nodes.append_columns(
    time=tables_other.nodes.time,
    flags=tables_other.nodes.flags,
    population=tables_other.nodes.population,
    individual=np.where(
        tables_other.nodes.individual != -1,
        tables_other.nodes.individual + individual_offset,
        -1,
    ),
    metadata=tables_other.nodes.metadata,
    metadata_offset=tables_other.nodes.metadata_offset,
)

# Append edges, remapping parent/child nodes using the offset
tables.edges.append_columns(
    left=tables_other.edges.left,
    right=tables_other.edges.right,
    parent=tables_other.edges.parent + node_offset,
    child=tables_other.edges.child + node_offset,
)

# Append migrations, remapping the associated node ID using the offset
tables.migrations.append_columns(
    left=tables_other.migrations.left,
    right=tables_other.migrations.right,
    source=tables_other.migrations.source,
    dest=tables_other.migrations.dest,
    time=tables_other.migrations.time,
    node=tables_other.migrations.node + node_offset,
)

tables.sort()

# Phase 2
print(f"\n PHASE 2: Simulating from {ISOLATION_TIME_YEARS:,} years ago backwards...")
print("  - Using the full OutOfAfrica_3G09 demography...")
print(f"  - Applying {MAP1_CHROM} recombination map to all ancestral lineages...")

ts_final = msprime.sim_ancestry(
    initial_state=tables,
    demography=msprime_demography_full,
    recombination_rate=rec_map_map1_sliced,
    start_time=t_iso_gen,
    record_migrations=True,
    record_full_arg=True,
    random_seed=30,
)

print("\n Adding neutral mutations to the final tree sequence...")
ts_mutated = msprime.sim_mutations(
    ts_final,
    rate=MUTATION_RATE,
    random_seed=40
)

print("\n Simulation done")
print("Summary of final tree sequence:")
print(ts_mutated)

tszip.compress(ts_mutated, OUT_DIRECTORY + "/" + OUT_NAME + ".trees.tsz")
