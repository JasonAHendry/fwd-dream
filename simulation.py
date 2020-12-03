import os
import errno
import sys
import datetime
import resource
import re
import time
import getopt
import configparser
import numpy as np
import pandas as pd
import scipy.stats
import random
import allel
from collections import Counter
import json

from lib.diagnostics import *
from lib.epochs import *
from lib.core import *
from lib.data_collection import *
from lib.generic import *


print("=" * 80)
print("Run forward-dream")
print("-" * 80)
print("Command: %s" % " ".join(sys.argv))
print("Run on host: %s" % os.uname().nodename)
print("Operating system: %s" % os.uname().sysname)
print("Machine: %s" % os.uname().machine)
print("Started at: %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 80)


# PARSE CLI INPUT
print("Parsing Command Line Inputs...")
migration = False
try:
    opts, args = getopt.getopt(sys.argv[1:], ":e:p:s:")
    # python simulation.py -e <expt-name> -p <param_set.ini> -s <balanced> -m <migration_dir>
    # Note -m is optional
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
            if e.errno != errno.EEXIST:
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
R0 = calc_R0(**equil_params)
bp_per_cM = 2 * params["nsnps"] / 100  # scaled for 1 CO per bivalent
# Options
options = {}
options['init_duration'] = eval(config.get('Options', 'init_duration'))
options['max_samples'] = eval(config.get('Options', 'max_samples'))
options['detection_threshold'] = eval(config.get('Options', 'detection_threshold'))

print("Parameter Values")
print("  No. of Humans:", params['nh'])
print("  No. of Vectors:", params['nv'])
print("  No. of Parasites per Human:", params['nph'])
print("  No. of Parasites per Vector:", params['npv'])
print("  No. Sites in Parasite Genome:", params['nsnps'])
print("  Bite Rate per Vector:", params['bite_rate_per_v'])
print("  Prob. Human Infected per Bite:", params['p_inf_h'])
print("  Prob. Vector Infected per Bite:", params['p_inf_v'])
print("  Prob. Parasite Strain Transmitted to Human:", params['p_k_h'])
print("  Prob. Parasite Strain Transmitted to Vector:", params['p_k_h'])
print("  Number of Oocysts Draw From Geometric with p =", params['p_oocysts'])
print("  Drift Rate in Humans:", params['drift_rate_h'])
print("  Drift Rate in Vectors:", params['drift_rate_v'])
print("  Mutation Rate in Humans (theta):", params['theta_h'])
print("  Mutation Rate in Vectors (theta):", params['theta_v'])
print("  bp per cM [Scaled for 1 CO / bivalent]:", bp_per_cM)
print("  Rate of Human Clearance (gamma):", params['gamma'])
print("  Rate of Vector Clearance (eta):", params['eta'])
print("  Lambda:", derived_params['lmbda'])
print("  Psi:", derived_params['psi'])
print("  Expected Human Prevalence:", x_h)
print("  Expected Vector Prevalence:", x_v)
print("  R0:", R0)
print("    Limit number of samples collected to:", options['max_samples'])
print("    Set the detection threshold for mixed infections to:", options['detection_threshold'])


# SAMPLING
print("Sampling Preferences")
prev_samp_freq = config.getint('Sampling', 'prev_samp_freq')
div_samp_freq = config.getint('Sampling', 'div_samp_freq')
report_rate = config.getint('Sampling', 'report_rate')

print("  Sampling Prevalence Every %d Days" % prev_samp_freq)
print("  Sampling Genetic Diversity Every %d Days" % div_samp_freq)
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
track_ibd = config.getboolean('Sampling', 'track_ibd')
print("    Track IBD?", track_ibd)
storage = DataCollection(prev_samp_freq=prev_samp_freq,
                         div_samp_freq=div_samp_freq,
                         max_samples=options['max_samples'],
                         detection_threshold=options['detection_threshold'],
                         track_ibd=track_ibd)


# DATA STRUCTURES
h = np.zeros(params['nh'], dtype=np.int8)  # Host infection status
v = np.zeros(params['nv'], dtype=np.int8)  # Vector infection status
h_dt = {}  # host index: parasite_genomes
v_dt = {}  # vector index: parasite genomes
t_h = np.zeros(params['nh'])  # Host time-of-last-update
t_v = np.zeros(params['nv'])  # Vector time-of-last-update
print("Done.")
print("")


# SEED INFECTION
n_seed = 10
h[:n_seed] = 1
if seed_method == "unique":
    # Each individual has a unique parasite genome, generated randomly
    h_dt.update({i : np.random.uniform(0, 1, size=(params["nph"], params["nsnps"])) for i in range(n_seed)})
elif seed_method == "clonal":
    # All individuals begin with the same genome
    h_dt.update({i : np.zeros((params["nph"], params["nsnps"])) for i in range(n_seed)})
else:
    raise Exception("'seed_method' not recognised. Choose from 'unique' or 'clonal'.")
h1 = np.sum(h == 1)
v1 = np.sum(v == 1)
print("Seeding simulation with %d infected humans" % h1)
print("...and %d infected vectors." % v1)
print("Done.")
print("")


# RUN
print("Running simulation...")
print("="*80)
print("Day: Number of days since the simulation began")
print("NH: Total number of hosts")
print("NV: Total number of vectors")
print("H1: Number of infected hosts")
print("V1: Number of infected vectors")
print("Hm: Number of hosts with a mixed infection")
print("Vm: Number of vectors with a mixed infection")
print("Elapsed (s): Seconds elapsed since last line was printed")
print("-"*80)
print("Day \t NH \t NV \t H1 \t V1 \t Hm \t Vm \t Elapsed (s) \t")
print("-"*80)
start_time = time.time()
trep = start_time  # time of last report
t0 = 0  # stores the current time, in days
tparams = 0  # stores the last time the simulation parameters changed, in days
gen = 0  # 'generations' of the simulation; one event (bite or clearance) occurs in each generation
history = {"inf_h": 0, "superinf_h": 0, "clear_h": 0, 
           "inf_v": 0, "superinf_v": 0, "clear_v": 0}  # keep track of events that occur
while t0 < max_t0:

    gen += 1
    
    """
    Epochs
    
    A simulation can pass through a series of Epochs
    each with a distinct set of simulation parameters.
    In the section below we move between Epochs and 
    optionally collect genetic data at Epoch boundaries,
    as specified by the parameter `.ini` file.
    
    """

    if any_epochs and t0 > init_t1:  # Completed initialisation
        for epoch in epochs:
            if t0 > epoch.t0 and not epoch.occurred:  # Entering Epoch
                epoch.occurred = True
                current_epoch = epoch
                print("~"*80)
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
                    v, t_v, v_dt = update_vectors(nv=params["nv"], v=v, t_v=t_v, v_dt=v_dt)
                    v1 = v.sum()  # recalculate number of infected vectors

                # Adjust sampling rates
                if current_epoch.adj_prev_samp:
                    base_prev_samp_freq = prev_samp_freq
                    storage.prev_samp_freq = current_epoch.prev_samp_freq
                    print("Adjusting Prevalence Sampling Rate:", base_prev_samp_freq)
                    print("... to:", storage.prev_samp_freq)

                if current_epoch.adj_div_samp:
                    base_div_samp_freq = div_samp_freq
                    storage.div_samp_freq = current_epoch.div_samp_freq
                    print("Adjusting Diversity Sampling Rate:", base_div_samp_freq)
                    print("... to:", storage.div_samp_freq)


                if current_epoch.calc_genetics or current_epoch.collect_samples:
                    # Bring host parasite population to present before collecting samples
                    h_dt = evolve_all_hosts(h_dt=h_dt, tis=t0-t_h, 
                                           drift_rate=params['drift_rate_h'], theta=params['theta_h'],
                                           nsnps=params['nsnps'])
                    t_h[:] = t0
                    
                    # Make an Epoch Directory
                    epoch_dir = os.path.join(out_path, current_epoch.name)
                    if not os.path.isdir(epoch_dir):
                        os.mkdir(epoch_dir)

                    if h1 > 0:
                        if current_epoch.collect_samples:
                            print("Collecting a sample of %d genomes..." % storage.max_samples)
                            ks, genomes, ixs = storage.collect_genomes(h_dt)
                            np.save(os.path.join(epoch_dir, "ks.npy"), ks)
                            np.save(os.path.join(epoch_dir, "genomes.npy"), genomes)
                            np.save(os.path.join(epoch_dir, "ixs.npy"), ixs)
                        if current_epoch.calc_genetics:
                            print("Storing genetic data upon Epoch entry...")
                            entry_genetics = storage.sample_genetics(t0=t0, h_dt=h_dt)
                            json.dump(entry_genetics, 
                                      open(os.path.join(epoch_dir, "entry_genetics.json"), "w"), 
                                      default=default)
                    else:
                        print("Human parasite population currently extinct!")
                        print("... can't compute genetics or collect genomes.")
                print("Done.")
                print("")
                        
                print("-"*80)
                print("Day \t NH \t NV \t H1 \t V1 \t Hm \t Vm \t Elapsed (s) \t")
                print("~"*80)

        # Subsequent `approach` updates occur here
        if current_epoch.t0 < t0 < current_epoch.approach_t1:
            if (t0 - tparams) > current_epoch.approach_delay:
                params.update(current_epoch.approach_params(t0))
                derived_params = calc_derived_params(params)
                equil_params = calc_equil_params(params, derived_params)

                if int(params["nv"]) != len(v):
                    v, t_v, v_dt = update_vectors(nv=params["nv"], v=v, t_v=t_v, v_dt=v_dt)
                    v1 = v.sum()  # recalculate number of infected vectors
                    
        # If the sampling rate changes during the Epoch, we adjust that here
        if current_epoch.adj_prev_samp:  # shut off adjusted sampling if...
            if storage.tprev > current_epoch.t0 + current_epoch.prev_samp_t:
                print("Returning to base prevalence sampling rate.")
                storage.prev_samp_freq = base_prev_samp_freq
                current_epoch.adj_prev_samp = False

        if current_epoch.adj_div_samp:  # shut off adjusted sampling if...
            if storage.tdiv > current_epoch.t0 + current_epoch.div_samp_t:
                print("Returning to base diversity sampling rate.")
                storage.div_samp_freq = base_div_samp_freq
                current_epoch.adj_div_samp = False


    """
    Sampling the parasite population
    
    Prevalence and genetic data is collected longitudinally
    during the simulation at a periodicity set by the user-
    defined parameters `prev_samp_freq` and `div_samp_freq`; 
    in the following section we collect information about 
    the present-day prevalence and genetic diversity of the 
    parasite population as required by these parameters.
    
    """
    
    # Sample prevalence
    if (t0 - storage.tprev) >= storage.prev_samp_freq:
        storage.sample_prevalence(t0=t0, h1=h1, v1=v1, nh=params['nh'], nv=params['nv'], 
                                  h_dt=h_dt, v_dt=v_dt)
    
    # Sample genetics
    if (t0 - storage.tdiv) >= storage.div_samp_freq:
        # Evolve hosts to sampling time
        h_dt = evolve_all_hosts(h_dt=h_dt, tis=t0-t_h, 
                                drift_rate=params['drift_rate_h'], theta=params['theta_h'],
                                nsnps=params['nsnps'])
        t_h[:] = t0

        # Sample genetics
        storage.sample_genetics(t0=t0, h_dt=h_dt)
                
    # Print a report to screen
    if gen % report_rate == 0:
        nHm = sum([storage.detect_mixed(genomes, options['detection_threshold']) for idh, genomes in h_dt.items()])
        nVm = sum([storage.detect_mixed(genomes, options['detection_threshold']) for idv, genomes in v_dt.items()])
        print("%.0f\t%d\t%d\t%d\t%d\t%d\t%d\t%.02f" \
              % (t0, 
                 params["nh"], params["nv"], 
                 h1, v1, 
                 nHm, nVm,
                 time.time() - trep))
        trep = time.time()
        
    """
    Moving forward in time
    
    The simulation progresses through time as a merged or parallel Poisson 
    process. Thus, the waiting times between events are exponentially  
    distributed with a rate parameter equal to the sum of the rates of the 
    individual event types:
    
        event_rate_total = sum(event_rate_1, ..., event_rate_n)
        
    We can split this process to only jump between events of interest,
    i.e. those that involve the transfer or clearance of malaria parasites.
    Since these events occur independently, the split processes are also
    poisson processes, and the waiting times between events is still
    exponentially distributed, but with a adjusted rates.
    
    We try to draw as few random numbers as possible to improve computational
    performance.
    
    """

    # Biting
    bite_rate = params['bite_rate_per_v']*params['nv']
    all_possible_biting_events = params['nv']*params['nh']
    infect_v = h1*(params['nv']-v1)*params['p_inf_v']
    infect_h = v1*(params['nh']-h1)*params['p_inf_h']
    superinfect_v = h1*v1*params['p_inf_v']
    superinfect_h = h1*v1*params['p_inf_h']
    superinfect_hv = h1*v1*params['p_inf_v']*params['p_inf_h']
    interesting_biting_events = np.array([infect_v,
                                          infect_h,
                                          superinfect_v,
                                          superinfect_h,
                                          superinfect_hv])
    interesting_biting_events *= bite_rate / all_possible_biting_events
    # Clearance
    clear_rate_h = params['gamma']*h1
    clear_rate_v = params['eta']*v1
    # Overall rate of events of interest
    rates = np.concatenate([interesting_biting_events, [clear_rate_h, clear_rate_v]])
    rates_total = rates.sum()  # -- will go to zero if extinction
    
    # Move forward in time to next event
    t0 += random.expovariate(rates_total)
    
    # Sample event type and simulate it
    u = random.random()
    p = rates.cumsum() / rates_total
    
    if u < p[0]:  # Infect vector
        idh = list(h_dt.keys())[int(random.random() * h1)]
        idv = random.choice(np.where(v==0)[0])
        
        # Evolve (h)          
        h_dt[idh] = evolve_host(hh=h_dt[idh], ti=t0-t_h[idh],
                                drift_rate=params['drift_rate_h'], theta=params['theta_h'],
                                nsnps=params['nsnps'])
        t_h[idh] = t0
        
        # Transfer (h -> v)
        v_dt[idv] = infect_vector(hh=h_dt[idh], vv=None, npv=params['npv'],
                                  nsnps=params['nsnps'],
                                  p_k=params['p_k_v'],
                                  p_oocysts=params['p_oocysts'],
                                  bp_per_cM=bp_per_cM)
        v[idv] = 1
        t_v[idv] = t0
        
        # Record
        history["inf_v"] += 1
    
    elif u < p[1]:  # Infect host
        idh = random.choice(np.where(h==0)[0])
        idv = list(v_dt.keys())[int(random.random() * v1)]
        
        # Evolve (v)       
        v_dt[idv] = evolve_vector(vv=v_dt[idv], ti=t0-t_v[idv],
                                  drift_rate=params['drift_rate_v'], theta=params['theta_v'],
                                  nsnps=params['nsnps'])
        t_v[idv] = t0
        
        # Transfer (v -> h)        
        h_dt[idh] = infect_host(hh=None, vv=v_dt[idv], nph=params['nph'], p_k=params['p_k_h'])
        h[idh] = 1
        t_h[idh] = t0
        
        # Record
        history["inf_h"] += 1
        
    elif u < p[2]:  # Superinfect vector
        idh = list(h_dt.keys())[int(random.random() * h1)]
        idv = list(v_dt.keys())[int(random.random() * v1)]
        
        # Evolve (h, v)          
        h_dt[idh] = evolve_host(hh=h_dt[idh], ti=t0-t_h[idh],
                                drift_rate=params['drift_rate_h'], theta=params['theta_h'],
                                nsnps=params['nsnps'])
        t_h[idh] = t0
        v_dt[idv] = evolve_vector(vv=v_dt[idv], ti=t0-t_v[idv],
                                  drift_rate=params['drift_rate_v'], theta=params['theta_v'],
                                  nsnps=params['nsnps'])
        t_v[idv] = t0
        
        # Transfer (h -> v)
        v_dt[idv] = infect_vector(hh=h_dt[idh], vv=v_dt[idv], npv=params['npv'],
                                  nsnps=params['nsnps'],
                                  p_k=params['p_k_v'],
                                  p_oocysts=params['p_oocysts'],
                                  bp_per_cM=bp_per_cM)
        
        # Record
        history["superinf_v"] += 1
    
    elif u < p[3]:  # Superinfect host
        idh = list(h_dt.keys())[int(random.random() * h1)]
        idv = list(v_dt.keys())[int(random.random() * v1)]
        
        # Evolve (h, v)          
        h_dt[idh] = evolve_host(hh=h_dt[idh], ti=t0-t_h[idh],
                               drift_rate=params['drift_rate_h'], theta=params['theta_h'],
                               nsnps=params['nsnps'])
        t_h[idh] = t0
        v_dt[idv] = evolve_vector(vv=v_dt[idv], ti=t0-t_v[idv],
                                  drift_rate=params['drift_rate_v'], theta=params['theta_v'],
                                  nsnps=params['nsnps'])
        t_v[idv] = t0
        
        # Transfer (v -> h)        
        h_dt[idh] = infect_host(hh=h_dt[idh], vv=v_dt[idv], nph=params['nph'], p_k=params['p_k_h'])
        t_h[idh] = t0
        
        # Record
        history["superinf_h"] += 1
    
    elif u < p[4]:  # Superinfect host and vector
        idh = list(h_dt.keys())[int(random.random() * h1)]
        idv = list(v_dt.keys())[int(random.random() * v1)]
        
        # Evolve (h, v)          
        h_dt[idh] = evolve_host(hh=h_dt[idh], ti=t0-t_h[idh],
                                drift_rate=params['drift_rate_h'], theta=params['theta_h'],
                                nsnps=params['nsnps'])
        t_h[idh] = t0
        v_dt[idv] = evolve_vector(vv=v_dt[idv], ti=t0-t_v[idv],
                                  drift_rate=params['drift_rate_v'], theta=params['theta_v'],
                                  nsnps=params['nsnps'])
        t_v[idv] = t0
        
        # Transfer (v -> h, h->v)
        h_dt[idh] = infect_host(hh=h_dt[idh], vv=v_dt[idv], nph=params['nph'], p_k=params['p_k_h'])
        v_dt[idv] = infect_vector(hh=h_dt[idh], vv=v_dt[idv], npv=params['npv'],
                                  nsnps=params['nsnps'],
                                  p_k=params['p_k_v'],
                                  p_oocysts=params['p_oocysts'],
                                  bp_per_cM=bp_per_cM)
        
        # Record
        history["superinf_v"] += 1
        history["superinf_h"] += 1
        
    elif u < p[5]:  # Host clears
        idh = list(h_dt.keys())[int(random.random() * h1)]
        h_dt.pop(idh)
        h[idh] = 0
        t_h[idh] = 0
        
        # Record
        history["clear_h"] += 1
        
    elif u < p[6]:  # Vector clears
        idv = list(v_dt.keys())[int(random.random() * v1)]
        v_dt.pop(idv)
        v[idv] = 0
        t_v[idv] = 0
        
        # Record
        history["clear_v"] += 1

    # Recalculate number of infected hosts and vectors
    h1 = h.sum()
    v1 = v.sum()
print("="*80)
print("Done.")
print("")

# Bring host parasite population to the present day
print("Evolving all host parasites to the present (day %.0f) ..." % t0)
h_dt = evolve_all_hosts(h_dt=h_dt, tis=t0-t_h, 
                       drift_rate=params['drift_rate_h'], theta=params['theta_h'],
                       nsnps=params['nsnps'])
t_h[:] = t0   
print("Done.")
print("")


# RUN DETAILS
print("Events simulated")
total = sum(history.values())
infection = history["inf_h"] + history["inf_v"]
superinfection = history["superinf_h"] + history["superinf_v"]
clearance = history["clear_h"] + history["clear_v"]
print("  Total: %d" % total)
print("  Infection: %d (%.02f%%)" % (infection, 100*infection/total))
print("    Host: %d (%.02f%%)" % (history["inf_h"], 100*history["inf_h"]/total))
print("    Vector: %d (%.02f%%)" % (history["inf_v"], 100*history["inf_v"]/total))
print("  Super-infection: %d (%.02f%%)" % (superinfection, 100*superinfection/total))
print("    Host: %d (%.02f%%)" % (history["superinf_h"], 100*history["superinf_h"]/total))
print("    Vector: %d (%.02f%%)" % (history["superinf_v"], 100*history["superinf_v"]/total))
print("  Clearance: %d (%.02f%%)" % (clearance, 100*clearance/total))
print("    Host: %d (%.02f%%)" % (history["clear_h"], 100*history["clear_h"]/total))
print("    Vector: %d (%.02f%%)" % (history["clear_v"], 100*history["clear_v"]/total))
print("")


# WRITE RESULTS
print("Writing Outputs...")
op = pd.DataFrame(storage.op)
og = pd.DataFrame(storage.og)
op.to_csv(os.path.join(out_path, "op.csv"), index=False)
og.to_csv(os.path.join(out_path, "og.csv"), index=False)

# Create Epoch DataFrame
if any_epochs:
    epoch_df = pd.DataFrame(np.zeros((len(epochs) + 1, 7)),
                            columns=["name", "t0", "t1", "gen_rate", "gens", "x_h", "x_v"])
    epoch_df.loc[0] = ["init", init_t0, init_t1, init_gen_rate, init_gens, x_h, x_v]
    for i, epoch in enumerate(epochs):
        epoch_df.loc[i + 1] = [epoch.name, epoch.t0, epoch.t1, epoch.gen_rate, epoch.gens, epoch.x_h, epoch.x_v]
    epoch_df.to_csv(os.path.join(out_path, "epoch_df.csv"), index=False)
print("Done.")
print("")

# Compute Genetics
print("Final Simulation State:")
print("*"*80)
if h1 > 0:
    final_prevalence = storage.sample_prevalence(t0=storage.tprev + storage.prev_samp_freq, 
                                                 h1=h1, v1=v1, 
                                                 nh=params['nh'], nv=params['nv'], 
                                                 h_dt=h_dt, 
                                                 v_dt=v_dt)
    final_genetics = storage.sample_genetics(t0=t0, h_dt=h_dt)
    
    for k, v in final_genetics.items():
        print("%s:\t%0.2f" % (k, v))         
else:
    print("Human parasite population currently extinct!")
    print("... can't compute genetics.")
print("*"*80)
print("Done.")
print("")

# Save run diagnostics
peak_memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 10**6
end_time = time.time()
runtime = str(datetime.timedelta(seconds=end_time - start_time))
print("Peak memory usage: %dMb" % peak_memory_mb)
print("Total simulation run-time (HH:MM:SS): %s" % runtime)
json.dump({"runtime": runtime, "peak_mem_mb": peak_memory_mb}, 
          open(os.path.join(out_path, "run_diagnostics.json"), "w"))

# Save events simulated
json.dump(history, open(os.path.join(out_path, "event_history.json"), "w"))

print("-" * 80)
print("Finished at: %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 80)

