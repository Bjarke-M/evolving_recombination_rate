import numpy as np
import os
import tskit
from collections import defaultdict
from tqdm import tqdm


def hot_recombination_map(recombination_rate, filename):
    """
    Write a hot (constant-rate) recombination map file for Relate.

    :param float recombination_rate: per-base per-generation recombination rate.
    :param str filename: output file path.
    """
    with open(filename, "w") as file:
        file.write("position COMBINED_rate(cM/Mb) Genetic_Map(cM)\n")
        file.write("0 " + '0.0' + " 0.0\n")
        file.write("499500 " + str(recombination_rate*1e8)  + " 0.0\n")
        file.write("500500 " + "0.0 " + str(recombination_rate*1000*1e2) + '\n')
        file.write("1000000 " + "0.0 " + str(recombination_rate*1000*1e2) + '\n')



def run_relate(
    ts,
    recombination_rate,
    mutation_rate,
    Ne,
    relate_loc,
    filename="relate",
    threads=1,
    memory=10,
):
    """
    Run the Relate pipeline on a simulated msprime tree sequence.

    Writes input files (VCF, ancestral FASTA, hot recombination map),
    runs Relate, and converts the output to tree sequence format.

    :param tskit.TreeSequence ts: input tree sequence from msprime.
    :param float recombination_rate: per-base per-generation recombination rate.
    :param float mutation_rate: per-base per-generation mutation rate.
    :param int Ne: effective population size (diploid). Converted to haploid internally.
    :param str relate_loc: path to the Relate installation directory.
    :param str filename: output file prefix (no extension).
    :param int threads: number of threads for Relate.
    :param int memory: total RAM in GB (divided by threads internally).
    """

    # --- Write input files ---
    print("Writing input files...", end="", flush=True)

    with open(filename + ".vcf", "w") as vcf_file:
        ts.write_vcf(vcf_file)

    ancestral_genome = ["A"] * int(ts.sequence_length)
    with open(filename + "_ancestral.fa", "w") as file:
        file.write(">ancestral_sequence\n")
        for s in ts.sites():
            ancestral_genome[int(s.position) - 1] = s.ancestral_state
        file.write("".join(ancestral_genome))
        file.write("\n")

    rec_map_file = filename + "_rec_map.txt"
    hot_recombination_map(recombination_rate, rec_map_file)

    print("done", flush=True)

    # --- Convert VCF to haps/sample format ---
    os.system(
        relate_loc
        + "/bin/RelateFileFormats --mode ConvertFromVcf --haps "
        + filename + ".haps --sample "
        + filename + ".sample -i "
        + filename
        + " > " + filename + ".relatelog 2>&1"
    )

    # --- Prepare input files (applies ancestral genome) ---
    os.system(
        relate_loc
        + "/scripts/PrepareInputFiles/PrepareInputFiles.sh --haps "
        + filename + ".haps --sample "
        + filename + ".sample --ancestor "
        + filename + "_ancestral.fa -o "
        + filename
        + " >> " + filename + ".relatelog 2>&1"
    )

    # --- Run Relate ---
    mem_per_thread = int(memory / threads)
    haploid_Ne = 2 * Ne

    if threads == 1:
        os.system(
            relate_loc
            + "/bin/Relate --mode All"
            + " --memory " + str(mem_per_thread)
            + " -m " + str(mutation_rate)
            + " -N " + str(haploid_Ne)
            + " --haps " + filename + ".haps"
            + " --sample " + filename + ".sample"
            + " --map " + rec_map_file
            + " --seed 1 -o " + filename
            + " >> " + filename + ".relatelog 2>&1"
        )
    else:
        os.system(
            relate_loc
            + "/scripts/RelateParallel/RelateParallel.sh"
            + " --memory " + str(mem_per_thread)
            + " -m " + str(mutation_rate)
            + " -N " + str(haploid_Ne)
            + " --haps " + filename + ".haps"
            + " --sample " + filename + ".sample"
            + " --map " + rec_map_file
            + " --seed 1 -o " + filename
            + " --threads " + str(threads)
            + " >> " + filename + ".relatelog 2>&1"
        )

    # --- Convert Relate output to tree sequence ---
    os.system(
        relate_loc
        + "/bin/RelateFileFormats --mode ConvertToTreeSequence -i "
        + filename + " -o " + filename + "_relate"
        + " >> " + filename + ".relatelog 2>&1"
    )


def load_relate_ts(filename="relate"):
    """
    Load the Relate output as a tskit tree sequence.

    :param str filename: the file prefix used in run_relate (no extension).
    :return: tskit.TreeSequence inferred by Relate.
    """
    return tskit.load(filename + "_relate.trees")



def map_mutations_to_region(ts, start_pos, end_pos, min_carriers=3):
    """
    Find all mutations whose carrier sets match a clade in the trees
    spanning [start_pos, end_pos].

    Collects mutations from the entire tree sequence that are topologically
    consistent with the local trees in the target region, regardless of
    where the mutations physically sit.

    Stage 1: Build a map from carrier set (frozenset of sample IDs) to
             the mutation site IDs that produced it.
    Stage 2: For each tree overlapping the region, check which carrier
             sets match a clade and collect the corresponding site IDs.

    :param tskit.TreeSequence ts: the input tree sequence.
    :param float start_pos: start of the genomic region.
    :param float end_pos: end of the genomic region.
    :param int min_carriers: minimum number of carriers for a mutation
        to be included (filters singletons/doubletons).
    :return: sorted list of unique site IDs mappable to the region.
    """

    # Stage 1: group mutations by carrier set
    print("Building map of mutations by carrier set...")
    mutations_by_carrier_set = defaultdict(list)
    for tree in tqdm(ts.trees(), total=ts.num_trees, desc="  Scanning mutations"):
        for mut in tree.mutations():
            carrier_samples = frozenset(tree.samples(mut.node))
            if min_carriers <= len(carrier_samples) <= ts.num_samples:
                mutations_by_carrier_set[carrier_samples].append(mut.site)
    print(f"  Found {len(mutations_by_carrier_set)} unique carrier sets.")

    # Stage 2: map carrier sets to clades in the target region
    print(f"Mapping mutations to trees in [{start_pos}, {end_pos}]...")
    mapped_sites = set()
    for tree in tqdm(ts.trees(), total=ts.num_trees, desc="  Mapping to region"):
        if tree.interval[1] <= start_pos or tree.interval[0] >= end_pos:
            continue
        # else:
        #     print(tree.index)
        for node in tree.nodes():
            if not tree.is_sample(node):
                clade_samples = frozenset(tree.samples(node))
                if clade_samples in mutations_by_carrier_set:
                    mapped_sites.update(mutations_by_carrier_set[clade_samples])

    print(f"  Mapped {len(mapped_sites)} unique sites to the region.")
    return sorted(mapped_sites)


def get_genotype_matrix(ts, site_ids):
    """
    Build a genotype matrix for a set of site IDs.

    Uses tskit.Variant to decode genotypes only for the requested sites.

    :param tskit.TreeSequence ts: the input tree sequence.
    :param list site_ids: list of site IDs (will be sorted internally).
    :return: a tuple (genotype_matrix, site_positions) where
        - genotype_matrix is an (m x n) numpy array (sites x samples).
        - site_positions is a list of m genomic positions.
    """
    if not site_ids:
        return np.array([]).reshape(0, ts.num_samples), []

    genotypes_list = []
    site_pos_list = []
    variant = tskit.Variant(ts)

    for site_id in sorted(site_ids):
        variant.decode(site_id)
        genotypes_list.append(np.array(variant.genotypes, copy=True))
        site_pos_list.append(int(variant.site.position))

    genotype_matrix = np.vstack(genotypes_list)
    return genotype_matrix, site_pos_list
