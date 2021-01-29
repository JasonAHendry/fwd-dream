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
from lib.preferences import *


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
try:
    opts, args = getopt.getopt(sys.argv[1:], ":e:p:s:c:")
    # Example
    # python simulation.py 
    # -e <expt_name> 
    # -p <params/param_file.ini> 
    # -s <clonal/unique>
    # -c eta=0.3
except getopt.GetoptError:
    print("Error parsing options.")
    
change_param = False
for opt, value in opts:
    if opt == "-e":
        expt_name = value
    elif opt == "-p":
        param_file = value
    elif opt == "-s":
        seed_method = value
    elif opt == "-c":
        change = {}
        change_param = True
        for p in value.strip().split(","):
            c = p.split("=")
            change[c[0]] = eval(c[1])
    else:
        print("Parameter %s not recognized." % opt)
        sys.exit(2)
        
# Create experiment directory
expt_path = os.path.join("results", expt_name)
try:
    os.mkdir(expt_path)
except OSError as e:  # handle race issues
    if e.errno != errno.EEXIST:
        raise

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
print("  Changing parameter(s)?:", change_param)
print("Done.")
print("")


# ASSIGN SIMULATION PARAMETERS
print("Loading Parameter File...")
config = configparser.ConfigParser()
config.read(param_file)

# Essential
params = parse_parameters(config)

# Change parameters if specified
if change_param:
    print("Changing %d simulation parameters via. CLI..." % len(change.keys()))
    for k, v in change.items(): print("  %s=%f" % (k, v))
    params.update(change)

# Derived & Equilibrium
derived_params = calc_derived_params(params)
equil_params = calc_equil_params(params, derived_params)
x_h = calc_x_h(**equil_params)
x_v = calc_x_v(**equil_params)
R0 = calc_R0(**equil_params)
bp_per_cM = 2 * params["nsnps"] / 100  # scaled for 1 CO per bivalent

# Options
options = {}
options['n_seed'] = eval(config.get('Options', 'n_seed'))
options['binary_genomes'] = eval(config.get('Options', 'binary_genomes'))
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
print("    Parasite genomes binary?:", options['binary_genomes'])


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
epochs = Epochs(params, config)
epochs.set_initialisation(verbose=True)
epochs.prepare_epochs(verbose=True)
max_t0 = epochs.max_t0  # pull this out of class for speed
print("Done.")
print("")


# STORAGE
print("Preparing Simulation Data Storage...")
track_ibd = config.getboolean('Sampling', 'track_ibd')
l_threshold = 2 * bp_per_cM if track_ibd else None  # for 2cM detection threshold
print("    Track IBD?", track_ibd)
if track_ibd:
    print("    IBD Length Threshold: 2cM = %dbp" % l_threshold)
storage = DataCollection(prev_samp_freq=prev_samp_freq,
                         div_samp_freq=div_samp_freq,
                         max_samples=options['max_samples'],
                         detection_threshold=options['detection_threshold'],
                         track_ibd=track_ibd, l_threshold=l_threshold)


# DATA STRUCTURES
h = np.zeros(params['nh'], dtype=np.int8)  # Host infection status
v = np.zeros(params['nv'], dtype=np.int8)  # Vector infection status
h_dt = {}  # host index: parasite genomes
v_dt = {}  # vector index: parasite genomes
t_h = np.zeros(params['nh'])  # Host time-of-last-update
t_v = np.zeros(params['nv'])  # Vector time-of-last-update
print("Done.")
print("")


# SEED INFECTION
print("Seeding simulation...")
n_seed = options['n_seed']
h[:n_seed] = 1
# Define data type of parasite genomes, and create clonally infected hosts
genome_dtype = np.int8 if options['binary_genomes'] else np.float32    
h_dt.update({i : np.zeros((params["nph"], params["nsnps"]), dtype=genome_dtype) 
             for i in range(n_seed)})
# Initial infection count    
h1 = np.sum(h == 1)
v1 = np.sum(v == 1)
print("  ...with %d infected humans" % h1)
print("  ...and %d infected vectors." % v1)
print("  Parasite genome data type: %s" % h_dt[0].dtype)
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
gen = 0  # 'generations' of the simulation; one event (bite or clearance) occurs in each generation
history = {"inf_h": 0, "superinf_h": 0, "clear_h": 0, 
           "inf_v": 0, "superinf_v": 0, "clear_v": 0}  # keep track of events that occur
extinct = False  # switches to True if parasite popn goes extinct
while t0 < max_t0:

    gen += 1  # possibly almost no longer necessary
    
    """
    Epochs
    
    A simulation can pass through a series of Epochs
    each with a distinct set of simulation parameters.
    In the section below we move between Epochs and 
    optionally collect genetic data at Epoch boundaries,
    as specified by the parameter `.ini` file.
    
    """

    if epochs.exist and t0 > epochs.init_duration:
        epochs.update_time(t0)
        
        # Entering new Epoch
        if not epochs.current.begun:
            epochs.current.begun = True
            print("~"*80)
            print("Beginning %s Epoch" % epochs.current.name)
            print("-"*80)
            print("Adjusting Parameter(s):", epochs.current.adj_keys)
            print("... to:", epochs.current.adj_vals)
            print("Using approach(es):", epochs.current.approach)
            print(" ... with an approach time of ",
                  epochs.current.approach_t1 - epochs.current.t0, " days.")
            print("Equilibrium host prevalence:", epochs.current.x_h)
            print("...and vector prevalence:", epochs.current.x_v)


            # Always update parameters at transition
            params.update(epochs.current.get_params(t0))
            derived_params = calc_derived_params(params)
            equil_params = calc_equil_params(params, derived_params)

            if int(params["nv"]) != len(v):
                v, t_v, v_dt = update_vectors(nv=params["nv"], v=v, t_v=t_v, v_dt=v_dt)
                v1 = v.sum()


            # Optionally update sampling rates
            if epochs.current.adj_prev_samp:
                print("Adjusting Prevalence Sampling Rate: %d" % storage.prev_samp_freq)
                print("... to: %d" % epochs.current.prev_samp_freq)
                storage.prev_samp_freq = epochs.current.prev_samp_freq

            if epochs.current.adj_div_samp:
                print("Adjusting Diversity Sampling Rate: %d" % storage.div_samp_freq)
                print("... to: %d" % epochs.current.div_samp_freq)
                storage.div_samp_freq = epochs.current.div_samp_freq


            if epochs.current.calc_genetics or epochs.current.save_state:
                # Bring host parasite population to present before collecting samples
                h_dt = evolve_all_hosts(h_dt=h_dt, tis=t0-t_h, 
                                       drift_rate=params['drift_rate_h'], theta=params['theta_h'],
                                       nsnps=params['nsnps'], binary_genomes=options['binary_genomes'])
                t_h[:] = t0

                # Make an Epoch Directory
                epoch_dir = os.path.join(out_path, epochs.current.name)
                if not os.path.isdir(epoch_dir):
                    os.mkdir(epoch_dir)

                if h1 > 0:
                    if epochs.current.save_state:
                        print("Saving simulation state...")
                        save_simulation(t0, h_dt, v_dt, t_h, t_v, epoch_dir)

                    if epochs.current.calc_genetics:
                        print("Storing genetic data upon Epoch entry...")
                        entry_genetics = storage.sample_genetics(t0=t0, h_dt=h_dt, 
                                                                 store=False, update=False)
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

                
        # During Epoch
        if epochs.current.adjust_params(t0):  # Check if parameters need updating
            params.update(epochs.current.get_params(t0))
            derived_params = calc_derived_params(params)
            equil_params = calc_equil_params(params, derived_params)

            if int(params["nv"]) != len(v):
                v, t_v, v_dt = update_vectors(nv=params["nv"], v=v, t_v=t_v, v_dt=v_dt)
                v1 = v.sum()  # recalculate number of infected vectors
        
        # Return to original sampling rates
        if epochs.current.adj_prev_samp:
            if t0 > epochs.current.t0 + epochs.current.prev_samp_t:
                storage.prev_samp_freq = prev_samp_freq
                epochs.current.adj_prev_samp = False
                
        if epochs.current.adj_div_samp:
            if t0 > epochs.current.t0 + epochs.current.div_samp_t:
                storage.div_samp_freq = div_samp_freq
                epochs.current.adj_div_samp = False


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
                                nsnps=params['nsnps'], binary_genomes=options['binary_genomes'])
        t_h[:] = t0

        # Sample genetics
        storage.sample_genetics(t0=t0, h_dt=h_dt)
                
    # Print a report to screen
    if gen % report_rate == 0:
        nHm = sum([storage.detect_mixed(genomes, options['detection_threshold']) 
                   for idh, genomes in h_dt.items()])
        nVm = sum([storage.detect_mixed(genomes, options['detection_threshold']) 
                   for idv, genomes in v_dt.items()])
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
    rates_total = rates.sum()
    
    if rates_total == 0:  # zero if extinction occurs
        print("Parasite population has gone extinct at day %d." % t0)
        print("Aborting simulation...")
        extinct = True
        break
        
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
                                nsnps=params['nsnps'], binary_genomes=options['binary_genomes'])
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
                                  nsnps=params['nsnps'], binary_genomes=options['binary_genomes'])
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
                                nsnps=params['nsnps'], binary_genomes=options['binary_genomes'])
        t_h[idh] = t0
        v_dt[idv] = evolve_vector(vv=v_dt[idv], ti=t0-t_v[idv],
                                  drift_rate=params['drift_rate_v'], theta=params['theta_v'],
                                  nsnps=params['nsnps'], binary_genomes=options['binary_genomes'])
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
                               nsnps=params['nsnps'], binary_genomes=options['binary_genomes'])
        t_h[idh] = t0
        v_dt[idv] = evolve_vector(vv=v_dt[idv], ti=t0-t_v[idv],
                                  drift_rate=params['drift_rate_v'], theta=params['theta_v'],
                                  nsnps=params['nsnps'], binary_genomes=options['binary_genomes'])
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
                                nsnps=params['nsnps'], binary_genomes=options['binary_genomes'])
        t_h[idh] = t0
        v_dt[idv] = evolve_vector(vv=v_dt[idv], ti=t0-t_v[idv],
                                  drift_rate=params['drift_rate_v'], theta=params['theta_v'],
                                  nsnps=params['nsnps'], binary_genomes=options['binary_genomes'])
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
                       nsnps=params['nsnps'], binary_genomes=options['binary_genomes'])
t_h[:] = t0   
print("Done.")
print("")

# WRITE RESULTS
print("Writing Outputs...")
op = pd.DataFrame(storage.op)
og = pd.DataFrame(storage.og)
op.to_csv(os.path.join(out_path, "op.csv"), index=False)
og.to_csv(os.path.join(out_path, "og.csv"), index=False)
epochs.write_epochs(out_path)

# Save parameter changes
if change_param:
    json.dump(change, open(os.path.join(out_path, "param_changes.json"), "w"))
    
# Save events simulated
json.dump(history, open(os.path.join(out_path, "event_history.json"), "w"))

# PRINT RUN DETAILS
print("Epidemiological Events Simulated")
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

# Compute Genetics
print("Final Population State")
if h1 > 0:
    print("  Prevalence Statistics")
    for metric, value in op.iloc[-1][1:].items():
        print("   %s: %.02f" % (op_names[metric], value))
    print("")
    
    print("  Genetic Diversity Statistics")
    for metric, value in og.iloc[-1][1:].items():
        statement = "    %s:"
        statement += " %d" if metric.startswith("n_") else " %.03f"
        print(statement % (genetic_names[metric], value))     
else:
    print("Host parasite population extinct!")
    print("... can't compute genetics.")
print("")

# Save run diagnostics
peak_memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 10**6
end_time = time.time()
runtime = str(datetime.timedelta(seconds=end_time - start_time))
print("Run Diagnostics")
print("  Extinction occurred?: %s" % extinct)
print("  Peak memory usage: %dMb" % peak_memory_mb)
print("  Total simulation run-time (HH:MM:SS): %s" % runtime)
json.dump({"runtime": runtime, "peak_mem_mb": peak_memory_mb, "extinct": extinct}, 
          open(os.path.join(out_path, "run_diagnostics.json"), "w"))
print("")

print("-" * 80)
print("Finished at: %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 80)

