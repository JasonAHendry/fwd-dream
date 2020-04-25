import os
import sys
from datetime import datetime
import re
import time
import getopt
import configparser
import numpy as np
import pandas as pd
import scipy.stats
import allel
from collections import Counter
import json

from lib.diagnostics import *
from lib.epochs import *
from lib.core import *
from lib.sequencing import *
from lib.generic import *


print("=" * 80)
print("Run forward-dream")
print("-" * 80)
print("Command: %s" % " ".join(sys.argv))
print("Run on host: %s" % os.uname().nodename)
print("Operating system: %s" % os.uname().sysname)
print("Machine: %s" % os.uname().machine)
print("Started at: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 80)


# PARSE CLI INPUT
print("Parsing Command Line Inputs...")
try:
    opts, args = getopt.getopt(sys.argv[1:], ":e:p:s:")
except getopt.GetoptError:
    print("Option Error. Please conform to:")
    print("-v <int>")
for opt, value in opts:
    if opt == "-e":
        expt_name = value
    elif opt == "-p":
        param_file = value
    elif opt == "-s":
        seed_method = value
    else:
        print("Parameter %s not recognized." % opt)
        sys.exit(2)
expt_path = os.path.join("results", expt_name)
if not os.path.isdir(expt_path):
    os.mkdir(expt_path)

# Create an output directory for the simulation: sim_<sim_suffix>_<sim_id>
while True:  # if running many parallel sims, race issues can arise in `out_path` assignment
    sim_dirs = os.listdir(expt_path)
    sim_suffix = "_".join(param_file.split("_")[1:]).split(".")[0]  # param_<sim_suffix>.ini
    sim_name = "sim_" + sim_suffix
    sim_counts = sum([1 for d in sim_dirs if sim_name in d])
    sim_id = "%04d" % (sim_counts + 1)
    out_dir = sim_name + "_" + sim_id
    out_path = os.path.join(expt_path, out_dir)
    if not os.path.isdir(out_path):
        try:  # e.g., another dir can get made after conditional but before os.mkdir()
            os.mkdir(out_path)
            break
        except OSError as e:
            if e.errno != os.errno.EEXIST:
                raise
            print("Directory assignment clash, trying again...")
            pass
print("  Experiment Directory:", expt_path)
print("  Parameter File:", param_file)
print("  Simulation Name:", sim_name)
print("  ...the %s simulation of this type." % sim_id)
print("  Output Directory:", out_dir)
print("  Seed Method:", seed_method)
print("Done.")
print("")

# ASSIGN SIMULATION PARAMETERS
print("Loading Parameter File...")
config = configparser.ConfigParser()
config.read(param_file)
# Basic
params = {}
demography = {param: int(val) for param, val in config.items('Demography')}
transmission = {param: float(val) for param, val in config.items('Transmission')}
genome = {param: int(val) for param, val in config.items('Genome')}
evolution = {param: float(val) for param, val in config.items('Evolution')}
params.update(demography)
params.update(transmission)
params.update(genome)
params.update(evolution)
# Derived & Equilibrium
derived_params = calc_derived_params(params)
equil_params = calc_equil_params(params, derived_params)
x_h = calc_x_h(**equil_params)
x_v = calc_x_v(**equil_params)
# Options
options = {}
options['init_duration'] = eval(config.get('Options', 'init_duration'))
options['back_mutation'] = config.getboolean('Options', 'back_mutation')
options['max_samples'] = eval(config.get('Options', 'max_samples'))
options['detection_threshold'] = eval(config.get('Options', 'detection_threshold'))
if seed_method == "seed_dir":
    options['seed_dir'] = config.get('Options', 'seed_dir')
    print("  Loading Seed Directory:", options['seed_dir'])

print("Parameter Values")
print("  No. of Humans:", params['nh'])
print("  No. of Vectors:", params['nv'])
print("  No. of Parasites per Human:", params['nph'])
print("  No. of Parasites per Vector:", params['npv'])
print("  No. Sites in Parasite Genome:", params['nsnps'])
print("  Bite Rate per Vector:", params['bite_rate_per_v'])
print("  Prob. Human Infected per Bite:", params['p_inf_h'])
print("  Prob. Vector Infected per Bite:", params['p_inf_v'])
print("  Drift Rate in Humans:", params['drift_rate_h'])
print("  Drift Rate in Vectors:", params['drift_rate_v'])
print("  Mutation Rate in Humans (theta):", params['theta_h'])
print("  Mutation Rate in Vectors (theta):", params['theta_v'])
print("  Rate of Human Clearance (gamma):", params['gamma'])
print("  Rate of Vector Clearance (eta):", params['eta'])
print("  Lambda:", derived_params['lmbda'])
print("  Psi:", derived_params['psi'])
print("  Expected Human Prevalence:", x_h)
print("  Expected Vector Prevalence:", x_v)
print("    Run with back mutation?", options['back_mutation'])
print("    Limit number of samples collected to:", options['max_samples'])
print("    Set the detection threshold for mixed infections to:", options['detection_threshold'])


# SAMPLING
# Rates
print("Sampling Preferences")
# parasite prevalence
prev_samp_strat = config.get('Sampling', 'prev_samp_rate')
if prev_samp_strat == 'variable':
    prev_samp_freq = config.getint('Sampling', 'prev_samp_freq')  # sample ~once every `_samp_freq` days
    print("  Sampling Prevalence Every %d Days" % prev_samp_freq)
# genetic diversity
div_samp_strat = config.get('Sampling', 'div_samp_rate')
if div_samp_strat == 'variable':
    div_samp_freq = config.getint('Sampling', 'div_samp_freq')
    print("  Sampling Genetic Diversity Every %d Days" % div_samp_freq)
# reports printed to screen
report_rate = config.getint('Sampling', 'report_rate')
print("  Printed Report Every %d Events" % report_rate)
print("Done.")
print("")


# PARSE EPOCHS
print("Preparing Epochs...")
# Initialization, until ~ simulation equilibrium
init_t0 = 0
if options['init_duration'] is None:
    print("Initializing to equilibrium.")
    time_to_equil = 2*x_h*params['nh']*(derived_params['h_v'] + derived_params['v_h'])  # 2*Ne*g
    print("  Init Duration (d):", time_to_equil)
    init_duration = time_to_equil
elif options['init_duration'] > 0:
    print("Initializing for a specified duration.")
    init_duration = options['init_duration']
    print("  Init Duration (d):", init_duration)
else:
    print("Inappropriate init_duration.")
init_t1 = init_t0 + init_duration
init_gen_rate = params['bite_rate_per_v']*params['nv'] \
                + params['gamma']*x_h*params['nh'] \
                + params['eta']*x_v*params['nv']
init_gens = init_duration*init_gen_rate  # days * gens per day = total gens
# Epochs
total_t0 = init_t1  # total time in simulation, given included epochs
total_gens = init_gens  # total gens in simulation, given included epochs
total_prev_samps = init_t1/prev_samp_freq
total_div_samps = init_t1/div_samp_freq
epoch_sections = [s for s in config.sections() if "Epoch" in s]
if len(epoch_sections) > 0:
    any_epochs = True
    epochs = [Epoch(config, s) for s in epoch_sections]
    print("Epochs")
    for (i, epoch) in enumerate(epochs):  # NB: correct order of epochs is assumed
        if i == 0:
            epoch.set_params(params)
            epoch.set_duration(init_t1)  # begins at end of initialization
            epoch.set_approach()
            epoch.set_sampling()
        else:
            epoch.set_params(epochs[i-1].epoch_params)
            epoch.set_duration(epochs[i-1].t1)  # begins at end `.t1` of previous epoch
            epoch.set_approach()
            epoch.set_sampling()
        total_prev_samps += epoch.calc_prev_samps(prev_samp_freq)
        total_div_samps += epoch.calc_div_samps(div_samp_freq)
        total_t0 += epoch.telapse
        total_gens += epoch.gens
        print(" ", i, ":", epoch.name)
        print("    Adjusting Parameter(s):", epoch.adj_keys)
        print("    To value(s):", epoch.adj_vals)
        print("    via.:", epoch.approaches)
        print("    Start: %.02f, Finish: %.02f" % (epoch.t0, epoch.t1))
        print("    Human Prevalence: %.03f, Vector: %.03f" % (epoch.x_h, epoch.x_v))
        print("    Generation Rate: %.04f, Total: %.04f" % (epoch.gen_rate, epoch.gens))
        print("    Adjust Prevalence Sampling:", epoch.adj_prev_samp)
        if epoch.adj_prev_samp:
            print("      to:", epoch.prev_samp_freq)
            print("      for:", epoch.prev_samp_t)
        print("    Adjust Diversity Sampling:", epoch.adj_div_samp)
        if epoch.adj_div_samp:
            print("      to:", epoch.div_samp_freq)
            print("      for:", epoch.div_samp_t)
else:
    any_epochs = False
# Totalling...
max_t0 = total_t0
max_gens = total_gens
prev_max_samps = int(total_prev_samps)
div_max_samps = int(total_div_samps)
print(" Total Generations:", max_gens)
print(" Total Duration:", max_t0)
print(" Total of %d Prevalence Samples" % prev_max_samps)
print(" Total of %d Diversity Samples" % div_max_samps)
print("Done.")
print("")


# STORAGE
print("Preparing Simulation Data Storage...")
# observe prevalence (op)
op = pd.DataFrame(np.zeros((prev_max_samps, 9)), columns=['t0', 'V1', 'VX', 'H1', 'HX', 'nHm', 'HmX', 'nVm', 'VmX'])
# observe genetics (og)
gen_metrics = ["n_re_genomes", "n_mixed_genomes",
               "n_re_samples", "n_mixed_samples",
               "mean_k", "n_barcodes", "single_barcodes",
               "n_variants", "n_segregating", "n_singletons", "pi", "theta", "tajd"]
track_ibd = config.getboolean('Sampling', 'track_ibd')
print("    Track IBD?", track_ibd)
if track_ibd:
    gen_metrics += ["avg_frac_ibd", "avg_n_ibd", "avg_l_ibd"]
og = pd.DataFrame(np.zeros((div_max_samps, 1 + len(gen_metrics))), columns=['t0'] + gen_metrics)
# optionally track site frequency spectrum
track_sfs = config.getboolean('Sampling', 'track_sfs')
print("    Track SFS?:", track_sfs)
if track_sfs:
    t0_sfs = np.zeros(div_max_samps)
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_midpoints = (bins[1:] + bins[:-1])/2.0
    binned_sfs_array = np.zeros((div_max_samps, n_bins), dtype='int16')
# optionally track r2
track_r2 = config.getboolean('Sampling', 'track_r2')
print("    Track r2?:", track_r2)
if track_r2:
    t0_r2 = np.zeros(div_max_samps)
    n_bins_r2 = 20
    bins_r2 = np.linspace(0, params['nsnps'], n_bins_r2 + 1)
    bin_midpoints_r2 = (bins_r2[1:] + bins_r2[:-1])/2.0
    binned_r2_array = np.zeros((div_max_samps, n_bins_r2))  # this should be floating point
# optionally track allele frequencies
track_allele_freqs = config.getboolean('Sampling', 'track_allele_freqs')
print("    Track Allele Frequencies?:", track_allele_freqs)
if track_allele_freqs:
    t0_freqs = np.zeros(div_max_samps)
    n_h_genomes = np.zeros(div_max_samps, dtype='int16')
    n_v_genomes = np.zeros(div_max_samps, dtype='int16')
    h_freqs = np.zeros((div_max_samps, params['nsnps']))
    v_freqs = np.zeros((div_max_samps, params['nsnps']))
track_params = config.getboolean('Sampling', 'track_allele_freqs')
print("    Track Parameter Changes?:", track_params)
if track_params:
    t0_params = []
    param_tracking = []
    t0_params.append(0)
    param_tracking.append(params.copy())  # record original parameters


# DATA STRUCTURES
v = np.zeros(params['nv'], dtype='int8')
h = np.zeros(params['nh'], dtype='int8')
h_a = np.zeros((params['nh'], params['nph'], params['nsnps']), dtype='int8')
v_a = np.zeros((params['nv'], params['npv'], params['nsnps']), dtype='int8')
t_h = np.zeros(params['nh'])
t_v = np.zeros(params['nv'])
print("Done.")
print("")


# SEED INFECTION
n_seed = 10
h[:n_seed] = 1  # indicate who is infected with binary array
v[:n_seed] = 1
if seed_method == "random":
    h_a[:n_seed] = np.random.choice([1, 2], size=(n_seed, params["nph"], params["nsnps"]))
    v_a[:n_seed] = np.random.choice([1, 2], size=(n_seed, params["nph"], params["nsnps"]))
elif seed_method == "balanced":
    seed_genome = np.random.choice([1, 2], size=params["nsnps"])
    h_a[:n_seed] = seed_genome
    v_a[:n_seed] = seed_genome
elif seed_method == "seed_dir":
    seed_dir = options["seed_dir"]
    seed_h_a = np.load(os.path.join(seed_dir, "h_a.npy"))
    seed_h = np.load(os.path.join(seed_dir, "h.npy"))
    print("Seeding from %s" % seed_dir)
    assert seed_h.sum() >= n_seed
    seed_ix = np.where(seed_h)[0]
    seed_h_ix = np.random.choice(seed_ix, n_seed, replace=False)
    seed_v_ix = np.random.choice(seed_ix, n_seed, replace=False)
    h_a[:n_seed] = seed_h_a[seed_h_ix]
    v_a[:n_seed] = seed_h_a[seed_v_ix]
else:
    h_a[:n_seed] = 1  # seed with parasite genomes
    v_a[:n_seed] = 1
h1 = np.sum(h == 1)
v1 = np.sum(v == 1)
print("Seeding simulation with %d infected humans" % h1)
print("...and %d infected vectors." % v1)
print("Done.")
print("")


# RUN
print("Running simulation...")
print("-"*80)
trep = time.time()  # time of last report
tparams = 0  # last time simulation parameters were changed
tprev = 0  # last time prevalence was recorded
tdiv = 0  # last time diversity was recorded
t0 = 0  # stores the current time, in days
gen = 0  # 'generations' of the simulation, one event (bite or clearance) occurs in each generation
prev_samp_ct = 0  # count of number of prevalence samples drawn
div_samp_ct = 0  # count of number of diversity samples drawn
while t0 < max_t0:

    gen += 1

    if any_epochs and t0 > init_t1:  # beyond initialization
        for epoch in epochs:
            if t0 > epoch.t0 and not epoch.occurred:
                epoch.occurred = True
                current_epoch = epoch
                print("-"*80)
                print("Beginning %s Epoch" % current_epoch.name)
                print("-"*80)
                print("Adjusting Parameter(s):", current_epoch.adj_keys)
                print("... to:", current_epoch.adj_vals)
                print("Using approach(es):", current_epoch.approaches)
                print(" ... with an approach time of ", current_epoch.approach_t1 - current_epoch.t0, " days.")
                print("Equilibrium host prevalence:", current_epoch.x_h)
                print("...and vector prevalence:", current_epoch.x_v)

                # Update parameters at transition, in case approach time is zero
                params.update(current_epoch.approach_params(t0))
                derived_params = calc_derived_params(params)
                equil_params = calc_equil_params(params, derived_params)

                if int(params["nv"]) != len(v):
                    params["nv"] = int(params["nv"])
                    if params["nv"] > len(v):
                        n_missing_v = params["nv"] - len(v)
                        v = np.concatenate((v, np.zeros(n_missing_v, dtype='int8')))
                        v_a = np.vstack((v_a, np.zeros((n_missing_v, params['npv'], params['nsnps']), dtype='int8')))
                        t_v = np.concatenate((t_v, np.zeros(n_missing_v)))
                    elif params["nv"] < len(v):
                        iv = np.random.choice(len(v), params["nv"], replace=False)  # vectors we keep
                        v = v[iv]
                        v_a = v_a[iv]
                        t_v = t_v[iv]

                # Adjust sampling rates
                if current_epoch.adj_prev_samp:
                    base_prev_samp_freq = prev_samp_freq
                    prev_samp_freq = current_epoch.prev_samp_freq

                if current_epoch.adj_div_samp:
                    base_div_samp_freq = div_samp_freq
                    div_samp_freq = current_epoch.div_samp_freq

                print("-"*80)

                if current_epoch.calc_genetics or current_epoch.collect_samples:
                    # To Present
                    h_a = evolve_all_hosts(h=h, h_a=h_a, tis=t0-t_h, 
                                           drift_rate=params['drift_rate_h'], theta=params['theta_h'],
                                           nsnps=params['nsnps'], back_mutation=options['back_mutation'])
                    t_h[:] = t0
                    v_a = evolve_all_vectors(v=v, v_a=v_a, tis=t0-t_v, 
                                             drift_rate=params['drift_rate_v'], theta=params['theta_v'],
                                             nsnps=params['nsnps'], back_mutation=options['back_mutation'])
                    t_v[:] = t0

                    epoch_dir = os.path.join(out_path, current_epoch.name)
                    if not os.path.isdir(epoch_dir):
                        os.mkdir(epoch_dir)

                    # In both cases, need to collect genomes...
                    if h1 > 0:
                        ks, genomes = collect_genomes(h, h_a,
                                                      max_samples=options['max_samples'],
                                                      detection_threshold=options['detection_threshold'])
                        if current_epoch.collect_samples:
                            print("Collecting genome samples.")
                            print("-"*80)
                            np.save(epoch_dir + "/genomes.npy", genomes)
                            np.save(epoch_dir + "/ks.npy", ks)
                        if current_epoch.calc_genetics:
                            print("Calculating genetic diversity.")
                            hap, pos, ac = gen_allel_datastructs(genomes)
                            k_stats = calc_k_stats(ks, verbose=True)
                            barcode_counts = get_barcodes(hap, verbose=True)
                            div_stats = calc_diversity_stats(pos, ac, verbose=True)
                            unfolded_sfs = calc_unfolded_sfs(ac, n=k_stats['n_re_genomes'], verbose=True)
                            pair_r2, pair_d = calc_r2_decay(hap, pos, ac, verbose=True)
                            # Write
                            np.save(epoch_dir + "/hap.npy", hap)
                            np.save(epoch_dir + "/ac.npy", ac)
                            np.save(epoch_dir + "/pair_r2.npy", pair_r2)
                            np.save(epoch_dir + "/pair_d.npy", pair_d)
                            np.save(epoch_dir + "/unfolded_sfs.npy", unfolded_sfs)
                            np.save(epoch_dir + "/barcode_counts.npy", barcode_counts)
                            json.dump(k_stats, open(epoch_dir + "/k_stats.json", "w"), default=default)
                            json.dump(div_stats, open(epoch_dir + "/div_stats.json", "w"), default=default)
                            print("-"*80)
                    else:
                        print("Human parasite population currently extinct!")
                        print("... can't compute genetics or collect genomes.")
                        print("-"*80)

                tparams = t0
                if track_params:
                    t0_params.append(t0)
                    param_tracking.append(params.copy())

        # Subsequent `approach` updates occur here
        if current_epoch.t0 < t0 < current_epoch.approach_t1:
            if (t0 - tparams) > current_epoch.approach_delay:
                params.update(current_epoch.approach_params(t0))
                derived_params = calc_derived_params(params)
                equil_params = calc_equil_params(params, derived_params)

                if int(params["nv"]) != len(v):
                    params["nv"] = int(params["nv"])
                    if params["nv"] > len(v):
                        n_missing_v = params["nv"] - len(v)
                        v = np.concatenate((v, np.zeros(n_missing_v, dtype='int8')))
                        v_a = np.vstack((v_a, np.zeros((n_missing_v, params['npv'], params['nsnps']), dtype='int8')))
                        t_v = np.concatenate((t_v, np.zeros(n_missing_v)))
                    elif params["nv"] < len(v):
                        iv = np.random.choice(len(v), params["nv"], replace=False)  # vectors we keep
                        v = v[iv]
                        v_a = v_a[iv]
                        t_v = t_v[iv]

                tparams = t0
                if track_params:
                    t0_params.append(t0)
                    param_tracking.append(params.copy())

    # Sample Prevalence
    if (t0 - tprev) > prev_samp_freq:
        if prev_samp_ct >= prev_max_samps:
            print("Not enough space to store sample!")
            samps_left = int((max_t0 - t0)/prev_samp_freq) + 1
            print("Expecting %d more samples" % samps_left)
            print("...adding space.")
            op = pd.concat([op, pd.DataFrame(np.zeros((samps_left, 9)),
                                             columns=['t0', 'V1', 'VX', 'H1', 'HX',
                                                      'nHm', 'HmX', 'nVm', 'VmX'])])
            op.index = list(range(len(op)))
            prev_max_samps += samps_left

        op.loc[prev_samp_ct]['t0'] = t0
        op.loc[prev_samp_ct]['V1'] = v1
        op.loc[prev_samp_ct]['VX'] = v1/float(params['nv'])
        op.loc[prev_samp_ct]['H1'] = h1
        op.loc[prev_samp_ct]['HX'] = h1/float(params['nh'])
        op.loc[prev_samp_ct]['nHm'] = sum([detect_mixed(indv, options['detection_threshold']) for indv in h_a])
        op.loc[prev_samp_ct]['HmX'] = op.loc[prev_samp_ct]['nHm']/float(params['nh'])
        op.loc[prev_samp_ct]['nVm'] = sum([detect_mixed(indv, options['detection_threshold']) for indv in v_a])
        op.loc[prev_samp_ct]['VmX'] = op.loc[prev_samp_ct]['nVm']/float(params['nv'])
        
        # Increment
        prev_samp_ct += 1
        tprev = t0
        if any_epochs and t0 > init_t1:
            if current_epoch.adj_prev_samp:  # shut off adjusted sampling if...
                if tprev > current_epoch.t0 + current_epoch.prev_samp_t:
                    print("Returning to base prevalence sampling rate.")
                    prev_samp_freq = base_prev_samp_freq
                    current_epoch.adj_prev_samp = False

    # Sample Genetics
    if (t0 - tdiv) > div_samp_freq:
        # To present
        h_a = evolve_all_hosts(h=h, h_a=h_a, tis=t0-t_h, 
                               drift_rate=params['drift_rate_h'], theta=params['theta_h'],
                               nsnps=params['nsnps'], back_mutation=options['back_mutation'])
        t_h[:] = t0
        v_a = evolve_all_vectors(v=v, v_a=v_a, tis=t0-t_v, 
                                 drift_rate=params['drift_rate_v'], theta=params['theta_v'],
                                 nsnps=params['nsnps'], back_mutation=options['back_mutation'])
        t_v[:] = t0

        if div_samp_ct >= div_max_samps:
            print("Not enough space to sample genetic diversity!")
            samps_left = int((max_t0 - t0)/div_samp_freq) + 1
            print("Expecting %d more genetic samples" % samps_left)
            print("...adding fridge space.")
            og = pd.concat([og, 
                            pd.DataFrame(np.zeros((samps_left, len(gen_metrics) + 1)), columns=['t0'] + gen_metrics)])
            og.index = list(range(len(og)))

            if track_sfs:
                t0_sfs = np.concatenate([t0_sfs, np.zeros(samps_left)])
                binned_sfs_array = np.vstack([binned_sfs_array, np.zeros(samps_left, n_bins)])

            if track_r2:
                t0_r2 = np.concatenate([t0_r2, np.zeros(samps_left)])
                binned_r2_array = np.vstack([binned_r2_array, np.zeros(samps_left, n_bins_r2)])

            if track_allele_freqs:
                t0_freqs = np.concatenate([t0_freqs, np.zeros(samps_left)])
                n_h_genomes = np.concatenate([n_h_genomes, np.zeros(samps_left)])
                n_v_genomes = np.concatenate([n_v_genomes, np.zeros(samps_left)])
                h_freqs = np.vstack([h_freqs, np.zeros((samps_left, params['nsnps']))])
                v_freqs = np.vstack([v_freqs, np.zeros((samps_left, params['nsnps']))])
            
            div_max_samps += samps_left
    
        # Compute Genetics
        if h1 > 0:
            # Collect a sample of genomes
            ks, genomes = collect_genomes(h, h_a, max_samples=options['max_samples'],
                                          detection_threshold=options['detection_threshold'])

            # Create haplotype, position, and allele count arrays
            hap, pos, ac = gen_allel_datastructs(genomes)

            # Compute and store summary statistics, SFS, r2
            og = store_genetics(ks=ks, hap=hap, pos=pos, ac=ac, 
                                og=og, t0=t0, div_samp_ct=div_samp_ct,
                                nsnps=params['nsnps'], track_ibd=track_ibd)
            if track_sfs: 
                t0_sfs, binned_sfs_array = store_sfs(ks=ks, ac=ac,
                                                     t0_sfs=t0_sfs, 
                                                     binned_sfs_array=binned_sfs_array,
                                                     t0=t0, div_samp_ct=div_samp_ct,
                                                     bins=bins) 
            if track_r2: 
                t0_r2, binned_r2_array = store_r2(ks=ks, hap=hap, pos=pos, ac=ac,
                                                  t0_r2=t0_r2, 
                                                  binned_r2_array=binned_r2_array,
                                                  t0=t0, div_samp_ct=div_samp_ct,
                                                  bins=bins_r2)

        # Track Allele Frequencies
        if track_allele_freqs:
            t0_freqs[div_samp_ct] = t0
            n_h_genomes[div_samp_ct] = h1*params['nph']
            n_v_genomes[div_samp_ct] = v1*params['npv']
            h_freqs[div_samp_ct] = h_a.sum((0, 1))
            v_freqs[div_samp_ct] = v_a.sum((0, 1))
        
        # Increment
        div_samp_ct += 1
        tdiv = t0
        if any_epochs and t0 > init_t1:
            if current_epoch.adj_div_samp:  # shut off adjusted sampling if...
                if tdiv > current_epoch.t0 + current_epoch.div_samp_t:
                    print("Returning to base diversity sampling rate.")
                    div_samp_freq = base_div_samp_freq
                    current_epoch.adj_div_samp = False

    # Print a `report` to screen
    if gen % report_rate == 0:
        print("t0: %.02f | V1: %d | H1: %d | nv: %d | bite_rate_per_v: %.02f |elapsed: %.02f" \
              % (t0, v1, h1, params["nv"], params['bite_rate_per_v'], time.time() - trep))
        trep = time.time()
        
    # Adjust population rates
    bite_rate = params['bite_rate_per_v']*params['nv']  # uninfected and infected bite
    clear_rate_h = params['gamma']*h1  # changes with h1
    clear_rate_v = params['eta']*v1
    rates = np.array((bite_rate, clear_rate_h, clear_rate_v))
    rates_total = rates.sum()
    
    # Move time forward until next event of type `typ`, based on population rates
    t0 += np.random.exponential(scale=1/rates_total)  # scale gives expectation
    typ = np.random.choice(3, size=1, p=rates/rates_total)
    
    if typ == 0:  # Bite
        idh = np.random.choice(params['nh'])
        idv = np.random.choice(params['nv'])
        
        if h[idh] == 1 and v[idv] == 1:
            # Evolve (h, v)
            h_a[idh] = evolve_host(hh=h_a[idh], ti=t0-t_h[idh],
                                   drift_rate=params['drift_rate_h'], theta=params['theta_h'],
                                   nsnps=params['nsnps'], back_mutation=options['back_mutation'])
            t_h[idh] = t0
            v_a[idv] = evolve_vector(vv=v_a[idv], ti=t0-t_v[idv],
                                     drift_rate=params['drift_rate_v'], theta=params['theta_v'],
                                     nsnps=params['nsnps'], back_mutation=options['back_mutation'])
            t_v[idv] = t0
            
            # Transfer (h -> v, v -> h)
            if np.random.uniform(0, 1) < params['p_inf_h']:
                h_a[idh] = infect_host(hh=h_a[idh], vv=v_a[idv])
            if np.random.uniform(0, 1) < params['p_inf_v']:  # sequential, so actually sampling from re-infected host
                v_a[idv] = infect_vector(hh=h_a[idh], vv=v_a[idv], nsnps=params['nsnps'])
        elif h[idh] == 0 and v[idv] == 1:
            # Evolve (v)       
            v_a[idv] = evolve_vector(vv=v_a[idv], ti=t0-t_v[idv],
                                     drift_rate=params['drift_rate_v'], theta=params['theta_v'],
                                     nsnps=params['nsnps'], back_mutation=options['back_mutation'])
            t_v[idv] = t0
            
            # Transfer (v -> h)
            if np.random.uniform(0, 1) < params['p_inf_h']:
                h_a[idh] = infect_host(hh=h_a[idh], vv=v_a[idv])
                h[idh] = 1 
                t_h[idh] = t0
        elif h[idh] == 1 and v[idv] == 0:
            # Evolve (h)          
            h_a[idh] = evolve_host(hh=h_a[idh], ti=t0-t_h[idh],
                                   drift_rate=params['drift_rate_h'], theta=params['theta_h'],
                                   nsnps=params['nsnps'], back_mutation=options['back_mutation'])
            t_h[idh] = t0
            
            # Transfer (h -> v)
            if np.random.uniform(0, 1) < params['p_inf_v']:
                v_a[idv] = infect_vector(hh=h_a[idh], vv=v_a[idv], nsnps=params['nsnps'])
                v[idv] = 1
                t_v[idv] = t0
    
    elif typ == 1:  # Host clears
        idh = np.random.choice(np.flatnonzero(h == 1)) if h1 > 1 else np.flatnonzero(h == 1)  # choice annoying default
        h[idh] = 0
        h_a[idh] = 0
        t_h[idh] = 0
    
    elif typ == 2:  # Vector clears
        idv = np.random.choice(np.flatnonzero(v == 1)) if v1 > 1 else np.flatnonzero(v == 1)
        v[idv] = 0
        v_a[idv] = 0
        t_v[idv] = 0

    h1 = h.sum()
    v1 = v.sum()
print("-"*80)
print("Done.")
print("")

# Clean extra rows
op = op.query("t0 > 0")
og = og.query("t0 > 0")
# To Present
print("Bringing all infected host and vector populations to the present (day %.0f) ..." % t0)
h_a = evolve_all_hosts(h=h, h_a=h_a, tis=t0-t_h, 
                       drift_rate=params['drift_rate_h'], theta=params['theta_h'],
                       nsnps=params['nsnps'], back_mutation=options['back_mutation'])
t_h[:] = t0
v_a = evolve_all_vectors(v=v, v_a=v_a, tis=t0-t_v, 
                         drift_rate=params['drift_rate_v'], theta=params['theta_v'],
                         nsnps=params['nsnps'], back_mutation=options['back_mutation'])
t_v[:] = t0       
print("Done.")
print("")




# WRITE RESULTS
print("Writing Outputs...")
op.to_csv(os.path.join(out_path, "op.csv"), index=False)
np.save(os.path.join(out_path, "h.npy"), h)
np.save(os.path.join(out_path, "h_a.npy"), h_a)
# adding a few parameters last min.
og["n_fixed_ref"] = params['nsnps'] - og["n_variants"]  # Total SNPs - Variant Sites (Incl. Fixed)
og["n_fixed_alt"] = og['n_variants'] - og["n_segregating"]  # Variant Sites (Incl. Fixed) - Variant (Not Fixed)
og["frac_mixed_genomes"] = og["n_mixed_genomes"] / og["n_re_genomes"]
og["frac_mixed_samples"] = og["n_mixed_samples"] / og["n_re_samples"]
og["frac_uniq_barcodes"] = og["single_barcodes"] / og["n_re_genomes"]
og.to_csv(os.path.join(out_path, "og.csv"), index=False)

if any_epochs:
    epoch_df = pd.DataFrame(np.zeros((len(epochs) + 1, 7)),
                            columns=["name", "t0", "t1", "gen_rate", "gens", "x_h", "x_v"])
    epoch_df.loc[0] = ["init", init_t0, init_t1, init_gen_rate, init_gens, x_h, x_v]
    for i, epoch in enumerate(epochs):
        epoch_df.loc[i + 1] = [epoch.name, epoch.t0, epoch.t1, epoch.gen_rate, epoch.gens, epoch.x_h, epoch.x_v]
    epoch_df.to_csv(os.path.join(out_path, "epoch_df.csv"), index=False)

if track_sfs:
    np.save(os.path.join(out_path, "t0_sfs.npy"), t0_sfs)
    np.save(os.path.join(out_path, "bin_midpoints.npy"), bin_midpoints)
    np.save(os.path.join(out_path, "binned_sfs_array.npy"), binned_sfs_array)

if track_r2:
    np.save(os.path.join(out_path, "t0_r2.npy"), t0_r2)
    np.save(os.path.join(out_path, "bin_midpoints_r2.npy"), bin_midpoints_r2)
    np.save(os.path.join(out_path, "binned_r2_array.npy"), binned_r2_array)

if track_allele_freqs:
    np.save(os.path.join(out_path, "t0_freqs.npy"), t0_freqs)
    np.save(os.path.join(out_path, "n_h_genomes.npy"), n_h_genomes)
    np.save(os.path.join(out_path, "n_v_genomes.npy"), n_v_genomes)                                    
    np.save(os.path.join(out_path, "h_freqs.npy"), h_freqs)
    np.save(os.path.join(out_path, "v_freqs.npy"), v_freqs)

if track_params:
    param_df = pd.concat([pd.DataFrame(param, index=[t0])
                          for t0, param in zip(t0_params, param_tracking)])
    param_df.to_csv(os.path.join(out_path, "param_df.csv"), index=True)
print("Done.")
print("")

# Compute Genetics
print("Final Simulation State:")
print("-"*80)
if h1 > 0:
    ks, genomes = collect_genomes(h, h_a,
                                  max_samples=options['max_samples'],
                                  detection_threshold=options['detection_threshold'])
    hap, pos, ac = gen_allel_datastructs(genomes)
    k_stats = calc_k_stats(ks, verbose=True)
    barcode_counts = get_barcodes(hap, verbose=True)
    div_stats = calc_diversity_stats(pos, ac, verbose=True)
    unfolded_sfs = calc_unfolded_sfs(ac, n=k_stats['n_re_genomes'], verbose=True)
    pair_r2, pair_d = calc_r2_decay(hap, pos, ac, verbose=True)
    # Write
    endpoint_dir = os.path.join(out_path, "Endpoint")
    if not os.path.isdir(endpoint_dir):
        os.mkdir(endpoint_dir)
    np.save(endpoint_dir + "/hap.npy", hap)
    np.save(endpoint_dir + "/ac.npy", ac)
    np.save(endpoint_dir + "/pair_r2.npy", pair_r2)
    np.save(endpoint_dir + "/pair_d.npy", pair_d)
    np.save(endpoint_dir + "/unfolded_sfs.npy", unfolded_sfs)
    np.save(endpoint_dir + "/barcode_counts.npy", barcode_counts)
    json.dump(k_stats, open(endpoint_dir + "/k_stats.json", "w"), default=default)
    json.dump(div_stats, open(endpoint_dir + "/div_stats.json", "w"), default=default)
else:
    print("Human parasite population currently extinct!")
    print("... can't compute genetics.")
print("-"*80)
print("Done.")
print("")

    
print("-" * 80)
print("Finished at: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 80)

