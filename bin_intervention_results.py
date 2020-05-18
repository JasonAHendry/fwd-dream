import os
import sys
import getopt
import configparser
from datetime import datetime
from lib.intervention import *
from lib.preferences import *

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


# FIND SIMULATION DIRECTORIES
print("Collecting all simulations under experiment...")
dirs = [os.path.join(expt_path, d) for d in os.listdir(expt_path)]
dirs = [d for d in dirs if os.path.isdir(d)]
complete_dirs = []
for d in dirs:
    contents = os.listdir(d)
    if "Endpoint" in contents:  # endpoint is only generated if there is no extinction
        complete_dirs.append(d)
print(" Found %d simulations." % len(dirs))
print(" %d are without extinction." % len(complete_dirs))

# LOAD SIMULATION EPOCHS AND PARAMETERS
epoch_df = pd.read_csv(os.path.join(complete_dirs[0], "epoch_df.csv"))  # assumes epochs are same for all replicates
epoch_df.to_csv(os.path.join(output_path, "epoch_df.csv"), index=False)  # drop this here for future use
max_t0 = epoch_df.t1.max()
config = configparser.ConfigParser()
config.read(param_path)
print("Done.")
print("")


# SPECIFY BINNING PREFERENCES
# TODO:
#  - workflow for if `_samp_strat` != 'variable'
obs_per_bin = 5
print("Set number of observations per bin to: %d" % obs_per_bin)
print("")


# BIN PREVALENCE DATA
# Determine bin size
print("Binning prevalence data...")
prev_samp_strat = config.get('Sampling', 'prev_samp_rate')
if prev_samp_strat == 'variable':
    prev_samp_freq = config.getint('Sampling', 'prev_samp_freq')  # sample ~once every `_samp_freq` days
    op_bin = obs_per_bin*prev_samp_freq  # Size of prevalence bin, in days

# Defining the bin boundaries
print("Defining bin boundaries...")
op_bins = np.arange(0, epoch_df.iloc[0].t1, op_bin)
for i, row in epoch_df[1:].iterrows():
    section = "Epoch_" + row['name']
    print("Preparing prevalence bins for", section)
    adj_prev = config.has_option(section, "prev_samp_freq")
    
    if adj_prev:
        print("  Bin size changed in epoch.")
        adj_prev_samp_freq = config.getfloat(section, "prev_samp_freq")
        adj_op_bin = adj_prev_samp_freq*obs_per_bin
        
        if config.has_option(section, "prev_samp_t"):
            adj_prev_samp_t = config.getint(section, "prev_samp_t")
            epoch_op_bins_adj = np.arange(row.t0, row.t0+adj_prev_samp_t, adj_op_bin)
            epoch_op_bins_base = np.arange(row.t0+adj_prev_samp_t, row.t1, op_bin)
            epoch_op_bins = np.concatenate([epoch_op_bins_adj, epoch_op_bins_base])
        else:
            epoch_op_bins = np.arange(row.t0, row.t1, adj_op_bin)  # adjust for full epoch
            op_bin = adj_op_bin  # update base rate if spans full epoch

    else:  # use base bin size
        epoch_op_bins = np.arange(row.t0, row.t1, op_bin)
    op_bins = np.concatenate([op_bins, epoch_op_bins])
op_bin_midpoints = (op_bins[:-1] + op_bins[1:]) / 2.0

# Load, bin, and average data
op_keep_cols = ["V1", "VX", "H1", "HX", "nHm", "HmX", "nVm", "VmX"]
ops = load_simulations(dirs=complete_dirs, file_name="op.csv")
ops_binned = bin_simulations(simulations=ops, bins=op_bins, keep_cols=op_keep_cols)
ops_array, ops_mean, ops_std, ops_se = average_simulations(simulations_binned=ops_binned,
                                                           keep_cols=op_keep_cols,
                                                           bin_midpoints=op_bin_midpoints)
# Write prevalence
print("Saving...")
np.save(output_path + "/ops_array.npy", ops_array)
ops_mean.to_csv(output_path + "/ops_mean.csv", index=False)
ops_std.to_csv(output_path + "/ops_stds.csv", index=False)
ops_se.to_csv(output_path + "/ops_se.csv", index=False)
print("Done.")
print("")


# BIN GENETIC DATA
# Load Data
print("Binning genetic data...")
# Determine bin size
div_samp_strat = config.get('Sampling', 'div_samp_rate')
if div_samp_strat == 'variable':
    div_samp_freq = config.getint('Sampling', 'div_samp_freq')  # sample ~once every `_samp_freq` days
    og_bin = obs_per_bin*div_samp_freq  # Size of prevalence bin, in days
# Define the bin boundaries
print("Defining bin boundaries...")
og_bins = np.arange(0, epoch_df.iloc[0].t1, og_bin)
for i, row in epoch_df[1:].iterrows():
    section = "Epoch_" + row['name']
    print("Preparing diversity bins for", section)
    adj_div = config.has_option(section, "div_samp_freq")
    
    if adj_div:
        print("  Bin size changed in epoch.")
        adj_div_samp_freq = config.getfloat(section, "div_samp_freq")
        adj_og_bin = adj_div_samp_freq*obs_per_bin
        
        if config.has_option(section, "div_samp_t"):
            adj_div_samp_t = config.getint(section, "div_samp_t")
            epoch_og_bins_adj = np.arange(row.t0, row.t0+adj_div_samp_t, adj_og_bin)
            epoch_og_bins_base = np.arange(row.t0+adj_div_samp_t, row.t1, og_bin)
            epoch_og_bins = np.concatenate([epoch_og_bins_adj, epoch_og_bins_base])
        else:
            epoch_og_bins = np.arange(row.t0, row.t1, adj_og_bin)
            og_bin = adj_og_bin
            
    else:  # use base bin size
        epoch_og_bins = np.arange(row.t0, row.t1, og_bin)
    og_bins = np.concatenate([og_bins, epoch_og_bins])
og_bin_midpoints = (og_bins[:-1] + og_bins[1:]) / 2.0
# Load, bin, and average data
og_keep_cols = genetic_metrics
track_ibd = config.getboolean('Sampling', 'track_ibd')
if track_ibd:
    og_keep_cols += ibd_metrics
ogs = load_simulations(dirs=complete_dirs, file_name="og.csv")
ogs_binned = bin_simulations(simulations=ogs, bins=og_bins, keep_cols=og_keep_cols)
ogs_array, ogs_mean, ogs_std, ogs_se = average_simulations(simulations_binned=ogs_binned,
                                                           keep_cols=og_keep_cols,
                                                           bin_midpoints=og_bin_midpoints)
# Write genetic data
print("Saving...")
np.save(output_path + "/ogs_array.npy", ogs_array)
ogs_mean.to_csv(output_path + "/ogs_mean.csv", index=False)
ogs_std.to_csv(output_path + "/ogs_stds.csv", index=False)
ogs_se.to_csv(output_path + "/ogs_se.csv", index=False)
print("Done.")
print("")


# BIN SITE-FREQUENCIES
track_sfs = config.getboolean('Sampling', 'track_sfs')
if track_sfs:
    print("Binning Site Frequency Spectra...")
    sfs_cols = np.load(os.path.join(complete_dirs[0], "bin_midpoints.npy"))  # they should be all the same
    sfss = load_simulations_from_npy(dirs=complete_dirs,
                                     file_name="binned_sfs_array.npy",
                                     times="t0_sfs.npy",
                                     columns=sfs_cols)
    sfss_binned = bin_simulations(simulations=sfss,
                                  bins=og_bins,  # we bin `sfs` the same as `og`, b/c collected at same frequency
                                  keep_cols=sfs_cols)
    _, sfss_mean, sfss_std, sfss_se = average_simulations(simulations_binned=sfss_binned,
                                                          keep_cols=sfs_cols,
                                                          bin_midpoints=og_bin_midpoints)
    # Prefer slightly different data format for future plotting
    sfs_t0 = np.array(sfss_mean.t0)
    sfs_bins = np.array(sfss_mean.columns[:-1], 'float')  # avoid t0
    sfs_mean = np.array(sfss_mean.drop("t0", 1), 'float')
    sfs_se = np.array(sfss_se.drop("t0", 1), 'float')
    # Write SFS data
    print("Saving...")
    np.save(os.path.join(output_path, "sfs_t0.npy"), sfs_t0)
    np.save(os.path.join(output_path, "sfs_mean.npy"), sfs_mean)
    np.save(os.path.join(output_path, "sfs_se.npy"), sfs_se)
    np.save(os.path.join(output_path, "sfs_bins.npy"), sfs_bins)
    print("Done.")
    print("")


# BIN LINKAGE-DECAY
track_r2 = config.getboolean('Sampling', 'track_r2')
if track_r2:
    print("Binning Linkage Decay...")
    r2_cols = np.load(os.path.join(complete_dirs[0], "bin_midpoints_r2.npy"))  # they should be all the same
    r2s = load_simulations_from_npy(dirs=complete_dirs,
                                    file_name="binned_r2_array.npy",
                                    times="t0_r2.npy",
                                    columns=r2_cols)
    r2s_binned = bin_simulations(simulations=r2s,
                                 bins=og_bins,  # we bin `r2` the same as `og`, b/c collected at same frequency
                                 keep_cols=r2_cols)
    _, r2s_mean, r2s_std, r2s_se = average_simulations(simulations_binned=r2s_binned,
                                                       keep_cols=r2_cols,
                                                       bin_midpoints=og_bin_midpoints)
    # Prefer slightly different data format for future plotting
    r2_t0 = np.array(r2s_mean.t0)
    r2_bins = np.array(r2s_mean.columns[:-1], 'float')  # avoid t0
    r2_mean = np.array(r2s_mean.drop("t0", 1), 'float')
    r2_se = np.array(r2s_se.drop("t0", 1), 'float')
    # Write r2 data
    print("Saving...")
    np.save(os.path.join(output_path, "r2_t0.npy"), r2_t0)
    np.save(os.path.join(output_path, "r2_mean.npy"), r2_mean)
    np.save(os.path.join(output_path, "r2_se.npy"), r2_se)
    np.save(os.path.join(output_path, "r2_bins.npy"), r2_bins)
    print("Done.")
    print("")
 
    
print("-" * 80)
print("Finished at: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 80)
    
