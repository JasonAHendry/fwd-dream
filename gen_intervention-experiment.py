"""
Generate BRMC submission files
To analyse the effect of a malaria
control intervention

Usage:
    python gen_intervention-experiment.py \
        -e <expt_name, str> \
        -p <params/param_file.ini, str> \
        -n <n_reps, int>


2020/01/27, JHendry
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
print("Generate submission script for intervention experiment")
print("-" * 80)
print("Command: %s" % " ".join(sys.argv))
print("Run on host: %s" % os.uname().nodename)
print("Operating system: %s" % os.uname().sysname)
print("Machine: %s" % os.uname().machine)
print("Started at: %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 80)

# PARSE CLI INPUT
print("Parsing Command Line Inputs...")

# Parse
try:
    opts, args = getopt.getopt(sys.argv[1:], ":e:p:n:")
except getopt.GetoptError:
    print("Error parising options.")
for opt, value in opts:
    if opt == "-e":
        expt_name = value
        print("  Experiment name: %s" % expt_name)
    elif opt == "-p":
        param_file = value
        print("  Parameter file: %s" % param_file)
    elif opt == "-n":
        n_reps = int(value)
        print("  Number of replicates: %d" % n_reps)
    else:
        print("  Option %s not recognized." % opt)
        sys.exit(2)
print("Done.")
print("")


# LOAD DEFAULT SIMULATION PARAMETERS
print("Loading Parameter File...")
config = configparser.ConfigParser()
config.read(param_file)
params = parse_parameters(config)
derived_params = calc_derived_params(params)
print("Done.")
print("")


# DEFINE EXPERIMENT SETTINGS
crash_x_h = 0.2
print("Host prevalence during crash: %.02f" % crash_x_h)

# Compute required parameter adjustments
crash_nv = nv_at_x_h(crash_x_h, params)
crash_gamma = gamma_at_x_h(crash_x_h, params, derived_params)
crash_br = bite_rate_per_v_at_x_h(crash_x_h, params, derived_params)

# Create modified parameter files
param_files_dt = {
    "param_artemisinin.ini": ["gamma", crash_gamma],
    "param_insecticide.ini": ["nv", crash_nv],
    "param_bednets.ini": ["bite_rate_per_v", crash_br],
}


# GENERATE PARAMETER AND SUBMISSION FILES
for fn, (key, val) in param_files_dt.items():
    print("Generating Parameter File: %s" % fn)
    print("  Changing parameter: %s" % key)
    print("  ..from %f" % params[key])
    print("  ..to %f" % val)
    
    # Modify configuration object
    print("  Updating parameter file...")
    # InitVar Epoch
    config.set("Epoch_InitVar", "adj_params", key)
    config.set("Epoch_InitVar", "adj_vals", "%f" % params[key])
    # Crash Epoch
    config.set("Epoch_Crash", "adj_params", key)
    config.set("Epoch_Crash", "adj_vals", "%f" % val)
    # CrashVar Epoch
    config.set("Epoch_CrashVar", "adj_params", key)
    config.set("Epoch_CrashVar", "adj_vals", "%f" % val)
    # Recovery Epoch
    config.set("Epoch_Recovery", "adj_params", key)
    config.set("Epoch_Recovery", "adj_vals", "%f" % params[key])
    
    # Write file
    output_path = os.path.join("params", fn)
    print("  Writing parameter file: %s" % output_path)
    with open(output_path, "w") as config_file:
        config.write(config_file)
        
    # Generating submission files
    submit_fn = "submit_%s-interventions.sh" % (key if key != "bite_rate_per_v" else "br")
    print("  Generating submission file: %s" % submit_fn)
    with open(submit_fn, "w") as f:
        f.write("#" + "="*80+"\n")
        f.write("# Submit a series of intervention experiments to BMRC\n")
        f.write("# Generated at: %s\n" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        f.write("# Replicates: %d\n" % n_reps)
        f.write("#" + "-"*80+"\n")
        f.write("\n")
        
        # Write submission statements
        for _ in range(n_reps):
            cmd = "qsub run_simulation.sh -e %s-%s -p %s -s clonal"
            f.write(cmd % (expt_name, key, output_path))
            f.write("\n")
        
    # Make executable
    os.chmod(submit_fn, 0o777)
        
    print("Done.")
    print("")


print("-" * 80)
print("Finished at: %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 80)

