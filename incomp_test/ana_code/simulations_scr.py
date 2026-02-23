#!/usr/bin/env python
# coding: utf-8

import stdpopsim
import msprime
import tskit
import tszip
import numpy as np
import random
import sys
import os

import simulations

#=======================================================================================================================

# Inputs
chromosome = sys.argv[1]  # e.g. chr12
moderns_Han = int(sys.argv[2])  # number of modern haploids to simulate for Han, Sardinian, Mbuti (each)
moderns_Sar = int(sys.argv[3])
moderns_Mbu = int(sys.argv[4])
mask = sys.argv[5]  # location of genomic mask to use with Relate (or "None")
# e.g. "/Users/ignatiev/Dropbox/projects/partc-26/data/mask"
rs = int(sys.argv[6])  # random seed to use for stdpopsim simulation
threads = int(sys.argv[7])  # for Relate: number of threads
memory = int(sys.argv[8])  # for Relate: *total* memory (will be divided by number of threads)

resimulate = False

# Update these to location of relate
# You can download this from
# https://github.com/MyersGroup/relate
# and follow the installation instructions
# relate_loc = "../relate"
relate_loc = "/Users/ignatiev/Desktop/relate"
rec_map_file = "/Users/ignatiev/Dropbox/projects/partc-26/data/genetic_maps/HapmapII_GRCh37/3col/genetic_map_GRCh37_" + chromosome + ".txt"
simulations_loc = "/Users/ignatiev/Dropbox/projects/partc-26/simulations/"

print("Inputs:")
print("chromosome:", chromosome)
print("modern samples:", moderns_Han, moderns_Sar, moderns_Mbu)
print("mask:", mask)
print("rs:", rs)
print("threads:", threads)
print("memory:", memory)

#=======================================================================================================================

if not resimulate:
    print("Checking if already have simulated trees...")
    # Check if we have already simulated this scenario
    with open(simulations_loc + "simulations_rs.csv", "r") as file:
        for line in file:
            line = line.strip().split(",")
            if int(line[0]) == moderns_Han and int(line[1]) == moderns_Sar and int(line[2]) == moderns_Mbu:
                rs = int(line[3])
                print("Simulation already done with seed", rs)
                break

if rs == -1:
    rs = random.randint(100000, 999999)

# We can use a genomic mask for Relate to mask centromeres, acrocentric p-arms, and regions with low
# sequencing quality, etc
# From Relate docs:
# Fasta file of same length as the ancestral genome containing a genomic mask.
# Loci passing the mask are denoted by P, loci not passing the mask are denoted by N. This is case insensitive.
if mask == "None":
    mask = None
else:
    # This is the 1000 Genomes Project pilot (strict) mask
    mask = mask + "/20140520." + chromosome + ".pilot_mask.fasta.gz"

# This defines the necessary stuff to simulate from the desired stdpopsim model
# https://popsim-consortium.github.io/stdpopsim-docs/stable/catalog.html#sec_catalog_HomSap
species = stdpopsim.get_species("HomSap")
model = species.get_demographic_model("AncientEurasia_9K19")
contig = species.get_contig(
    chromosome=chromosome,
    genetic_map="HapMapII_GRCh37",
    mutation_rate=model.mutation_rate,
)
mutation_rate = model.mutation_rate
recombination_map = contig.recombination_map
Ne = 27000  # diploid, prior on population size

print("-"*80)
print("Simulation parameters:")
print("chromosome:", chromosome)
print("mutation rate:", model.mutation_rate)
print("avg recombination rate: " + str(recombination_map.mean_rate), flush=True)
print("ploidy:", contig.ploidy)
print(" ")

# This is the number of sequences from each population to sample
# "time" defines the sampling time, in generations
# (model generation time is 25 years)
samples = [
    msprime.SampleSet(moderns_Mbu, population="Mbuti", time=0, ploidy=2),
    msprime.SampleSet(moderns_Sar, population="Sardinian", time=0, ploidy=2),
    msprime.SampleSet(moderns_Han, population="Han", time=0, ploidy=2),
    # msprime.SampleSet(1, population="MA1", time=960, ploidy=1),
    # msprime.SampleSet(1, population="UstIshim", time=1800, ploidy=1),
    # msprime.SampleSet(1, population="Neanderthal", time=2000, ploidy=1),
]
print("-"*80)

# Name of files to save the outputs to
filename = chromosome + "_" + str(rs) + "_" + str(moderns_Han) + "_" + str(moderns_Sar) + "_" + str(moderns_Mbu) + "_sim"

#=======================================================================================================================

# Simulating trees

if os.path.exists(filename + ".trees"):
    print("Already have simulated trees, running Relate only")
    ts = tskit.load(filename + ".trees")
elif os.path.exists(simulations_loc + filename + ".trees"):
    print("Already have simulated trees, running Relate only")
    ts = tskit.load(simulations_loc + filename + ".trees")
elif os.path.exists(filename + ".trees.tsz"):
    print("Already have simulated trees, running Relate only")
    ts = tszip.decompress(filename + ".trees.tsz")
elif os.path.exists(simulations_loc + filename + ".trees.tsz"):
    print("Already have simulated trees, running Relate only")
    ts = tszip.decompress(simulations_loc + filename + ".trees.tsz")
else:
    ts, rs = simulations.simulate_data(
        contig,
        model,
        samples,
        rs,
    )
    tszip.compress(ts, filename + ".trees.tsz")

    print("\nRecording sample ages...", flush=True)
    sample_ages = np.zeros(ts.num_samples)
    with open(filename + "_sample_info.csv", "w") as file:
        file.write("sample,population,sampling_time\n")
        for k, pop in enumerate(ts.populations()):
            print(pop.metadata["id"], pop.metadata["description"], pop.metadata["sampling_time"], end=" ")
            if len(ts.samples(k)) > 0:
                for i in ts.samples(k):
                    sample_ages[i] = ts.node(i).time
                    print(ts.node(i).time, end=" ")
                    file.write(
                        str(i)
                        + ","
                        + str(pop.metadata["id"])
                        + ","
                        + str(ts.node(i).time)
                        +"\n"
                    )
                print(" ")
    np.savetxt(filename + "_sample_ages.txt", sample_ages, delimiter="\n")
    print(" ")

print("Writing poplabels...", end="", flush=True)
simulations.write_poplabels(ts, filename)
print("done")

print("-"*80)

#=======================================================================================================================

# Running Relate

print("Running Relate")
simulations.run_relate(
    ts,
    rec_map_file,
    mutation_rate,
    2 * Ne,  # need HAPLOIDS
    relate_loc,
    mask=mask,
    sample_ages=False,
    memory=int(memory/threads),
    threads=threads,
    filename=filename,
    filename_poplabels=filename + ".poplabels",
    pop_of_interest="Han,Mbuti,Sardinian",
)

# Tidying up and converting output to tree sequence (ARG) format

os.system(
    "gzip "
    + filename
    + ".anc"
)
os.system(
    "gzip "
    + filename
    + ".mut"
)
os.system(
    relate_loc
    + "/bin/RelateFileFormats --mode ConvertToTreeSequence -i "
    + filename
    + " -o "
    + filename + "_relate"
    + " >> " + filename + ".relatelog 2>&1"
)
os.system(
    "tszip " + filename + "_relate.trees" + " >> " + filename + ".relatelog 2>&1"
)
os.system(
    relate_loc
    + "/bin/RelateFileFormats --mode ConvertToTreeSequence -i "
    + filename + ".popsize"
    + " -o "
    + filename + "_relate_popsize"
    + " >> " + filename + ".relatelog 2>&1"
)
os.system(
    "tszip " + filename + "_relate_popsize.trees >> " + filename + ".relatelog 2>&1"
)
# os.system(
#     "rm " + filename + "_ancest.haps; "
#     + "rm " + filename + "_ancestral.fa; "
#     + "rm " + filename + ".haps; "
#     + "rm " + filename + ".sample; "
#     + "rm " + filename + ".vcf; "
# )
os.system(
    "mv " + filename + "*" + " " + simulations_loc
)

with open(simulations_loc + "simulations_rs.csv", "w") as file:
    file.write(str(moderns_Han) + "," + str(moderns_Sar) + "," + str(moderns_Mbu) + "," + str(rs) + "\n")
