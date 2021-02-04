"""
Bin a set of forward-dream malaria control intervention
experiments
JHendry, 2021/02/04

This is necessary to compute average trajectories of
statistics across simulations, as forward-dream operates 
in continuous time.

Usage:
  python bin_intervention-experiment.py \
    -e <expt_name> \
    -p <params/param_file.ini>
    
    
The script will output to /analysis/<expt_name>.


"""


import os
import sys
import getopt
import configparser
import json
from datetime import datetime
from lib.temporal_binning import *


print("=" * 80)
print("Bin and average results from forward-dream simulation")
print("-" * 80)
print("Command: %s" % " ".join(sys.argv))
print("Run on host: %s" % os.uname().nodename)
print("Operating system: %s" % os.uname().sysname)
print("Machine: %s" % os.uname().machine)
print("Started at: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 80)


# PARSE COMMAND-LINE
print("Parsing Command Line Inputs...")
try:
    opts, args = getopt.getopt(sys.argv[1:], ":e:p:")
except getopt.GetoptError:
    print("Option Error. Please conform to:")
    print("-e <str> -p <str>")

for opt, value in opts:
    if opt == "-e":
        expt_name = value
        expt_path = os.path.join("results", expt_name)
        output_path = os.path.join("analysis", expt_name)
        print("Preparing to bin data for", expt_name)
        print(" Output path:", output_path)
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
    elif opt == "-p":
        param_path = value
        print(" Parameter path:", param_path)
    else:
        print("Parameter %s not recognized." % opt)
        sys.exit(2)
print("Done.")
print("")


# SEARCH FOR SIMULATIONS
print("Searching for all experiment simulations...")
dirs = [os.path.join(expt_path, d) for d in os.listdir(expt_path)]
dirs = [d for d in dirs if os.path.isdir(d)]
complete_dirs = []
for d in dirs:
    contents = os.listdir(d)
    run_diagnostics = json.load(open(os.path.join(d, "run_diagnostics.json"), "r"))
    if not run_diagnostics["extinct"]:
        complete_dirs.append(d)
print(" Found %d total simulations." % len(dirs))
print("   Extinction occurred in: %d" % (len(dirs) - len(complete_dirs)))
print("Done.")
print("")

# Load Epochs
print("Loading Epoch data frame and parameter file...")
epoch_df = pd.read_csv(os.path.join(complete_dirs[0], "epoch_df.csv"))  # assumes epochs are same for all replicates
epoch_df.to_csv(os.path.join(output_path, "epoch_df.csv"), index=False)  # drop this here for future use
print(epoch_df)
epoch_df.index = epoch_df.name

# Load Parameters
config = configparser.ConfigParser()
config.read(param_path)
print("Done.")
print("")


# BIN DATA
# Preferences
print("Setting binning preferences...")
obs_per_bin = 5  # the number of observations to include in each bin, on average
print("  Observations per bin: %d" % obs_per_bin)
print("Done.")
print("")

# GENETIC DIVERSITY DATA
print("Binning genetic diversity data...")
# Create boundaries for bins
print("  Creating bin boundaries...")
og_bounds = create_bin_boundaries(config=config, 
                                  epoch_df=epoch_df, 
                                  data_type="div", 
                                  obs_per_bin=obs_per_bin)
# Load all the diversity simulations
print("  Loading data frames...")
ogs = load_simulations(dirs=complete_dirs, 
                       file_name="og.csv")
# Bin them in time
print("  Binning data frames...")
keep_cols = [c for c in ogs[0].columns if c != "t0"]
ogs_binned = bin_simulations(simulations=ogs, 
                             bins=og_bounds, 
                             keep_cols=keep_cols)

# Compute means, standard devation, standard error across simulations
print("  Computing summary statistics...")
ogs_array, ogs_mean, ogs_std, ogs_se = average_simulations(simulations_binned=ogs_binned,
                                                           keep_cols=keep_cols,
                                                           bin_midpoints=(og_bounds[1:] + og_bounds[:-1])/2)

# Write outputs
print("  Writing...")
np.save(output_path + "/ogs_array.npy", ogs_array)
ogs_mean.to_csv(output_path + "/ogs_mean.csv", index=False)
ogs_std.to_csv(output_path + "/ogs_stds.csv", index=False)
ogs_se.to_csv(output_path + "/ogs_se.csv", index=False)
print("Done.")
print("")


# PREVALENCE DATA
print("Binning prevalence data...")
# Create boundaries for bins
print("  Creating bin boundaries...")
op_bounds = create_bin_boundaries(config=config, 
                                  epoch_df=epoch_df, 
                                  data_type="prev", 
                                  obs_per_bin=obs_per_bin)
# Load all the diversity simulations
print("  Loading data frames...")
ops = load_simulations(dirs=complete_dirs, 
                       file_name="op.csv")
# Bin them in time
print("  Binning data frames...")
keep_cols = [c for c in ops[0].columns if c != "t0"]
ops_binned = bin_simulations(simulations=ops, 
                             bins=op_bounds, 
                             keep_cols=keep_cols)

# Compute means, standard devation, standard error across simulations
print("  Computing summary statistics...")
ops_array, ops_mean, ops_std, ops_se = average_simulations(simulations_binned=ops_binned,
                                                           keep_cols=keep_cols,
                                                           bin_midpoints=(op_bounds[1:] + op_bounds[:-1])/2)
                             
# Write prevalence
print("  Writing...")
np.save(output_path + "/ops_array.npy", ops_array)
ops_mean.to_csv(output_path + "/ops_mean.csv", index=False)
ops_std.to_csv(output_path + "/ops_stds.csv", index=False)
ops_se.to_csv(output_path + "/ops_se.csv", index=False)
print("Done.")
print("")


print("-" * 80)
print("Finished at: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 80)
