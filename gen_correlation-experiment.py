"""
Generate BMRC submission files for fwd-dream to conduct 
an experiment exploring how genetic diversity statistics 
are correlated with parasite prevalence

2020/12/23, JHendry
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
run_sensitivity = False

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
        run_sensitivity = True
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
derived_params = calc_derived_params(params)
print("Done.")
print("")


# LOAD SENSITIVITY FILE
if run_sensitivity:
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
    
    # Draw from uniform rv?
    draw_uniform = False
    print(" Draw from uniform distribution:", draw_uniform)
    
    print("Done.")
    print("")
else:
    print("Not performing sensitivity analysis.")


# WRITE SUBMISSION FILE
# Define experiment settings
print("Defining experiment settings...")
prevs = np.arange(0.1, 0.9, 0.1)
print(" Prevalence range:", prevs)
print("Done.")

# COMPUTE NUMBER OF SIMULATIONS TO SUBMIT
print("Number of experiments: %d" % 3)  # nv, gamma, br
print("Number of replicates: %d" % n_reps)
print("Number of prevalence levels: %d" % len(prevs))
print("Sensitivity analysis?", run_sensitivity)
if run_sensitivity:
    n_param_combos = sum([len(v) + 1 for k, v in sensitivity.items()])
    print("  Parameter combinations: %d" % n_param_combos)
else:
    n_param_combos = 1
n_total = 3*n_reps*len(prevs)*n_param_combos
print("TOTAL: %d" % n_total)

# Prepare submission files
print("Preparing submission files...")
f_br = open("submit_br-correlations.sh", "w")
f_gamma = open("submit_gamma-correlations.sh", "w")
f_nv = open("submit_nv-correlations.sh", "w")

for f in [f_br, f_gamma, f_nv]:
    f.write("="*80+"\n")
    f.write("Submit a series of correlation experiments to BMRC\n")
    f.write("Generated at: %s\n" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    f.write("Total: %d\n" % (n_total/3))
    f.write("-"*80+"\n")
    f.write("\n")
    
# Prepare a template for the submission statement
statement = "qsub run_simulation.sh"
statement += " -e %s"  # will change depending on br, gamma, nv
statement += " -p %s" % param_file
statement += " -s clonal"

# Iterate over all parameter combinations
print("Writing submission statements for all parameter combinations...")
for p in prevs:
    print("="*80)
    print("Host Prevalence: %.02f" % p)
    
    # Calc. adjusments to achieve prevalence
    nv = nv_at_x_h(x_h=p, params=params)

    gamma = gamma_at_x_h(x_h=p, params=params, 
                         derived_params=derived_params)

    br = bite_rate_per_v_at_x_h(x_h=p, params=params,
                                derived_params=derived_params)
    
    # Write submission statments
    dt = {
        "nv": (f_nv, nv),
        "gamma": (f_gamma, gamma),
        "bite_rate_per_v": (f_br, br)
    }
    
    for name, (f, adj) in dt.items():
        for _ in range(n_reps):
            f.write(statement % (expt_name + "-" + name))
            f.write(" -c %s=%s" % (name, adj))
            f.write("\n")
    
    if run_sensitivity:
        print("Running sensitivity analysis at this prevalence level...")
        for s, vals in sensitivity.items():
            print("  Perturb: %s" % s)
            print("    Range:", vals)
            
            if draw_uniform:
                print("Error: drawing uniform not built.")
                sys.exit(2)
            else:
                for v in vals:
                    # Copy & perturb parameters
                    params_changed = params.copy()
                    params_changed.update({s: v})
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
                    
                    # Write submission statements
                    for name, (f, adj) in dt.items():
                        for _ in range(n_reps):
                            f.write(statement % (expt_name + "-" + name))
                            f.write(" -c %s=%s,%s=%s" % (s, v, name, adj))
                            f.write("\n")
    print("Done.")
    print("")
    print("-"*80)

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
    
 
