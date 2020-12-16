"""
Generate BMRC submission files for fwd-dream to conduct 
an experiment exploring how genetic diversity statistics 
are correlated with parasite prevalence

2020/12/16, JHendry
"""


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
import json

from lib.diagnostics import *
from lib.epochs import *
from lib.generic import *


print("=" * 80)
print("Generate submission script for correlation experiment")
print("Including a sensitivity analysis")
print("-" * 80)
print("Command: %s" % " ".join(sys.argv))
print("Run on host: %s" % os.uname().nodename)
print("Operating system: %s" % os.uname().sysname)
print("Machine: %s" % os.uname().machine)
print("Started at: %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 80)


# PARSE CLI INPUT
print("Parsing Command Line Inputs...")

# Defaults
change_param = False

# Parse
try:
    opts, args = getopt.getopt(sys.argv[1:], ":e:p:s:n:")
except getopt.GetoptError:
    print("Error parising options.")
for opt, value in opts:
    if opt == "-e":
        expt_name = value
        print("  Experiment name: %s" % expt_name)
    elif opt == "-p":
        param_file = value
        print("  Parameter file: %s" % param_file)
    elif opt == "-s":
        sens_file = value
        print("  Sensitivity file: %s" % sens_file)
    elif opt == "-n":
        n_reps = int(value)
        print("  Number of replicates: %d" % n_reps)
    else:
        print("  Option %s not recognized." % opt)
        sys.exit(2)
print("Done.")
print("")
        
    
# Create experiment directory
expt_path = os.path.join("results", expt_name)
if not os.path.isdir(expt_path):
    os.mkdir(expt_path)

    
# LOAD DEFAULT SIMULATION PARAMETERS
print("Loading Parameter File...")
config = configparser.ConfigParser()
config.read(param_file)
params = parse_parameters(config)


# LOAD SENSITIVITY FILE
print("Loading Sensitivity File...")
config = configparser.ConfigParser()
config.read(sens_file)

# Instantiate sensitivity dictionary
sensitivity = {}

# Extract parameters from config
demography = {param: [int(v) for v in val.strip().split(",")]
              for param, val in config.items('Demography')}
transmission = {param: [float(v) for v in val.strip().split(",")]
                for param, val in config.items('Transmission')}
evolution = {param: [float(v) for v in val.strip().split(",")]
             for param, val in config.items('Evolution')}

# Construct dictionary
sensitivity.update(demography)
sensitivity.update(transmission)
sensitivity.update(evolution)


# WRITE SUBMISSION FILE
# Define experiment settings
print("Experiment settings...")
draw_uniform = False
prevs = np.arange(0.1, 0.9, 0.1)
print(" Draw from uniform distribution:", draw_uniform)
print(" Prevalence range:", prevs)

# Compute total number of experiments that will be run
n_params_perturbed = len(sensitivity.keys())
n_param_combos = sum([len(v) + 1 for k, v in sensitivity.items()]) # parameter combinations
n_expt_types = 3
n_prev = len(prevs)
print("Experiment statistics")
print("  No. parameters perturbed: %d" % n_params_perturbed)
print("  No. parameter combinations: %d" % n_param_combos)
print("  No. experiment types: %d" % n_expt_types)
print("  No. prevalence values: %d" % n_prev)
print("  No. replicates: %d" % n_reps)
print("  TOTAL: %d" % (n_param_combos*n_expt_types*n_prev*n_reps))

# Prepare submission files
f_br = open("submit_br-correlations.sh", "w")
f_gamma = open("submit_gamma-correlations.sh", "w")
f_nv = open("submit_nv-correlations.sh", "w")

for f in [f_br, f_gamma, f_nv]:
    f.write("="*80+"\n")
    f.write("Submit a series of correlation experiments to BMRC\n")
    f.write("-"*80+"\n")
    f.write("\n")
    
# Prepare a template for the submission statement
statement = "qsub run_simulation.sh"
statement += " -e %s"  # will change depending on br, gamma, nv
statement += " -p %s" % param_file
statement += " -s clonal"

# Now we iterate over all perturbations
print("Perturbing parameters and writing...")
for s, vals in sensitivity.items():
    print("-"*80)
    print("Perturbing parameter: %s" % s)
    print("")
    print("High:", vals[1])
    print("Default:", params[s])
    print("Low:", vals[0])
    print("")
    
    # Perturbatio method
    if draw_uniform:
        pass
    else:
        vals.append(params[s]) # append the default value
        
        for v in vals:
            qsub = statement
            qsub += " -c %s=%s" % (s, str(v))
            
            for p in prevs:
                
                # Copy & perturb parameters
                params_changed = params.copy()  # make a copy of the base parameter file
                params_changed.update({s: v})  # change a parameter
                derived_params_changed = calc_derived_params(params_changed)
                
                # Calc. adjusments to achieve prevalence
                nv = nv_at_x_h(x_h=p,
                               params=params_changed)
                
                gamma = gamma_at_x_h(x_h=p, 
                                     params=params_changed, 
                                     derived_params=derived_params_changed)
                
                br = bite_rate_per_v_at_x_h(x_h=p, 
                                            params=params_changed,
                                            derived_params=derived_params_changed)
                
                # Write submission statments
                dt = {
                    "nv": [f_nv, nv],
                    "gamma": [f_gamma, gamma],
                    "br": [f_br, br]
                }
                
                for name, l in dt.items():
                    f, adj = l[0], l[1]
                    for _ in range(n_reps):
                        f.write(qsub % (expt_name + "-" + name))  # correct expt name
                        f.write(",%s=%s\n" % (name, adj))

# Close all files
for f in [f_br, f_gamma, f_nv]:
    f.write("\n")
    f.write("="*80+"\n")
    f.close()
print("Done.")
print("")

print("-" * 80)
print("Finished at: %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 80)   
    
 