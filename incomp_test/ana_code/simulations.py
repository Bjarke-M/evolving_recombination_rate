import stdpopsim
import random
import os

#=======================================================================================================================

def simulate_data(contig, model, samples, rs=-1):
    """
    Function to simulate a tree sequence using stdpopsim
    :param contig: stdpopsim contig object
    :param model:  stdpopsim demographic model name
    :param samples: list of msprime SampleSet objects
    :param rs: random seed to use
    :return:
    """
    if rs == -1:
        rs = 1 + random.randrange(1000000)
        print("setting random seed:", rs)
    engine = stdpopsim.get_engine("msprime")

    print("Simulating...", end="", flush=True)
    ts = engine.simulate(model, contig, samples, random_seed=rs, record_provenance=False)
    print("done")

    print("random seed:", rs)
    print("sequence interval:", ts.first().interval[0], ts.last().interval[1])
    print("sequence length: ", ts.sequence_length)
    print("number of trees: ", ts.num_trees)
    print("number of mutations: ", ts.num_mutations)
    print("number of samples:", ts.num_samples)

    return ts, rs


def write_poplabels(ts, filename):
    """
    Write a file with the population labels for each sample
    :param ts: input tree sequence
    :param filename: name of output file
    :return:
    """
    with open(filename + ".poplabels", "w") as file:
        file.write("sample population group sex\n")
        for s in ts.individuals():
            node = ts.node(s.nodes[0])
            pop = ts.population(node.population).metadata["id"]
            file.write("tsk_" + str(s.id) + " " + pop + " " + pop + " NA\n")


def run_relate(
    ts,
    rec_map_file,
    mutation_rate,
    Ne,
    relate_loc,
    mask=None,
    sample_ages=False,
    memory=10,
    threads=1,
    filename = "relate",
    filename_poplabels = "relate",
    pop_of_interest = "",
):
    """
    Runs Relate on the input tree sequence
    :param ts: input tree sequence to extract data from
    :param rec_map_file: recombination map
    :param mutation_rate: mutation rate per site per generation
    :param Ne: effective population size of HAPLOIDS
    :param relate_loc: location of relate_loc
    :param mask: location of genetic mask to use
    :param sample_ages: whether to use sample ages
    :param memory: RAM per thread
    :param threads: how many threads for Relate
    :param filename: output filehandle (no extension)
    :param filename_poplabels: output filehandle for the .poplabels file (no extension)
    :param pop_of_interest: populations to estimate sizes for

    :return:
    """

    print("Writing files...", end="", flush=True)
    with open(filename + ".vcf", "w") as vcf_file:
        ts.write_vcf(vcf_file)

    if mask is not None:
        mask = " --mask " + mask
    else:
        mask = ""

    ancestral_genome = ["A"] * int(ts.sequence_length)
    with open(filename + "_ancestral.fa", "w") as file:
        file.write(">ancestral_sequence\n")
        for s in ts.sites():
            ancestral_genome[int(s.position) - 1] = s.ancestral_state
        file.write("".join(ancestral_genome))
        file.write("\n")

    print("done", flush=True)

    # Convert vcf to haps, sample format
    os.system(
        relate_loc
        + "/bin/RelateFileFormats --mode ConvertFromVcf --haps "
        + filename
        + ".haps --sample "
        + filename
        + ".sample -i "
        + filename
        + " > " + filename + ".relatelog 2>&1"
    )

    # Run script to prepare input files
    os.system(
        relate_loc
        + "/scripts/PrepareInputFiles/PrepareInputFiles.sh --haps "
        + filename
        + ".haps --sample "
        + filename
        + ".sample --ancestor " + filename + "_ancestral.fa -o "
        + filename
        + mask
        + " >> " + filename + ".relatelog 2>&1"
    )

    sample_ages_string = ""
    if sample_ages:
        sample_ages_string = "--sample_ages " + filename + "_sample_ages.txt "

    # Run Relate
    if threads == 1:
        os.system(
            relate_loc
            + "/bin/Relate --mode All "
            + sample_ages_string
            + "--memory "
            + str(memory)
            + " -m "
            + str(mutation_rate)
            + " -N "
            + str(Ne)
            + " --haps "
            + filename
            + ".haps --sample "
            + filename
            + ".sample --map "
            + str(rec_map_file)
            + " --seed 1 -o "
            + filename
            + " >> " + filename + ".relatelog 2>&1"
        )
    else:
        os.system(
            relate_loc
            + "/scripts/RelateParallel/RelateParallel.sh "
            + sample_ages_string
            + "--memory "
            + str(memory)
            + " -m "
            + str(mutation_rate)
            + " -N "
            + str(Ne)
            + " --haps "
            + filename
            + ".haps --sample "
            + filename
            + ".sample --map "
            + str(rec_map_file)
            + " --seed 1 -o "
            + filename
            + " --threads "
            + str(threads)
            + " >> " + filename + ".relatelog 2>&1"
        )

    if pop_of_interest != "":
        pop_of_interest = " --pop_of_interest " + pop_of_interest

    os.system(
        relate_loc
        + "/scripts/EstimatePopulationSize/EstimatePopulationSize.sh "
        + "-i "
        + filename
        + " -m "
        + str(mutation_rate)
        + " --poplabels "
        + filename_poplabels
        + pop_of_interest
        + " --seed 1 -o "
        + filename + ".popsize"
        + " --threads "
        + str(threads)
        + " >> " + filename + ".relatelog 2>&1"
    )


def flat_recombination_map(recombination_rate, sequence_length, filename):
    with open(filename, "w") as file:
        file.write("position COMBINED_rate(cM/Mb) Genetic_Map(cM)\n")
        file.write("0 " + str(recombination_rate * 1e8) + " 0.0\n")
        file.write(
            str(int(sequence_length))
            + " 0.0 "
            + str(recombination_rate * 1e2 * sequence_length)
            + "\n"
        )

