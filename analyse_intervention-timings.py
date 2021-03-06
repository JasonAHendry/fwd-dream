"""
Compute the 'response times' for prevalence and genetic diversity metrics
for a forward-dream malaria control intervention experiment

Statistics are computed on individual simulations.

Usage:
  python analyse_intervention-timings.py \
    -e <exp_name>
    
The script will output to `analysis/<expt_name>/response`


"""


import os
import sys
import getopt
import json
from datetime import datetime
import pandas as pd
import numpy as np
from lib.preferences import *
from lib.response import *
from lib.plotting import *


print("=" * 80)
print("Calculate response time statistics following an intervention experiment with `forward-dream`")
print("-" * 80)
print("Command: %s" % " ".join(sys.argv))
print("Run on host: %s" % os.uname().nodename)
print("Operating system: %s" % os.uname().sysname)
print("Machine: %s" % os.uname().machine)
print("Started at: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 80)




# PARSE USER INPUTS
print("Parsing user inputs...")
try:
    opts, args = getopt.getopt(sys.argv[1:], ":e:")
except getopt.GetoptError:
    print("Option error. Check command line arguments.")
    sys.exit(2)

for opt, value in opts:
    if opt == "-e":
        expt_name = value
        expt_path = os.path.join("results", expt_name)
        print("  Experiment:", expt_name)
        print("  Input directory:", expt_path)
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
    try:
        run_diagnostics = json.load(open(os.path.join(d, "run_diagnostics.json"), "r"))
        if not run_diagnostics["extinct"]:
            complete_dirs.append(d)
        else:
            print("  %s experienced extinction." % d)
    except FileNotFoundError:
        print("  %s is not completed." % d)
print("  Found %d total simulations." % len(dirs))
print("    No. incomplete/extinct: %d" % (len(dirs) - len(complete_dirs)))
sim_complete = [os.path.basename(d) for d in complete_dirs]
n_sims = len(sim_complete)
print("    No. complete simulations: %d" % n_sims)
output_path = os.path.join("analysis", expt_name, "response")
if not os.path.exists(output_path):
    os.makedirs(output_path)
print("  Output directory:", output_path)
print("Done.")
print("")




# PREFERENCES
savefig = True
analysis_metrics = ['HX', 'VX', 
                    'frac_mixed_samples','mean_k',
                    'n_singletons','n_segregating','pi','theta',
                    'f_ibd',  # I have excluded `l_ibd`, because it does not change enough 
                    'f_ibs', 'l_ibs']
genetic_names.update({"mean_k": "C.O.I. ($k$)",
                      "pi": "Nucl. Diversity ($\pi$)"})
response_dt = {}



# CALCULATE RESPONSE TIMES FOR CRASH
print("Calculating response times for crash...")
ds = []
es = []
print("-"*80)
for sim in sim_complete:
    print("%s..." % sim)
    # Load full data frame
    epoch_df = pd.read_csv(os.path.join(expt_path, sim, "epoch_df.csv"))
    epoch_df.index = epoch_df.name
    og = pd.read_csv(os.path.join(expt_path, sim, "og.csv"))
    op = pd.read_csv(os.path.join(expt_path, sim, "op.csv"))

    # Merge on time
    ot = pd.merge(left=op, right=og, on="t0")
    
    # Compute statistics
    d = calc_detection_time(ot, epoch_df, 
                            initial_epoch="InitVar", 
                            respond_epoch="Crash", 
                            robustness=3, alpha=0.01,
                            analysis_metrics=analysis_metrics)
    
    e = calc_equilibrium_time(ot, epoch_df,
                              initial_epoch="InitVar", 
                              respond_epoch="Crash",
                              equilibrium_epoch="CrashVar",
                              robustness=6,
                              analysis_metrics=analysis_metrics)
    
    # Store
    ds.append(d)
    es.append(e)
print("-"*80)
print("Aggregating...")
detect_df = pd.concat(ds, 1).transpose()
equil_df = pd.concat(es, 1).transpose()
# Store medians
response_dt["metric"] = detect_df.columns
response_dt["crash_detect"] = detect_df.median().values
response_dt["crash_equil"] = equil_df.median().values
print("Saving...")
detect_df.to_csv(os.path.join(output_path, "crash_detection.csv"), index=False)
equil_df.to_csv(os.path.join(output_path, "crash_equilibrium.csv"), index=False)
print("Done.")
print("")




# VISUALIZE SELECTED FOR CRASH
print("Visualizing randomly selected set...")
n_view = 5
view = np.random.choice(len(sim_complete), n_view, replace = False)
example_path = os.path.join(output_path, "examples")
if not os.path.exists(example_path):
    os.makedirs(example_path)
print("  #: %d" % n_view)
print("  to: %s" % example_path)

for ix in view:
    
    # Select individual
    sim = sim_complete[ix]
    
    # Load full data frame
    epoch_df = pd.read_csv(os.path.join(expt_path, sim, "epoch_df.csv"))
    epoch_df.index = epoch_df.name
    og = pd.read_csv(os.path.join(expt_path, sim, "og.csv"))
    op = pd.read_csv(os.path.join(expt_path, sim, "op.csv"))

    # Merge on time
    ot = pd.merge(left=op, right=og, on="t0")

    # Compute statistics
    d = calc_detection_time(ot, epoch_df, 
                            initial_epoch="InitVar", 
                            respond_epoch="Crash", 
                            robustness=3, alpha=0.01,
                            analysis_metrics=analysis_metrics)

    e = calc_equilibrium_time(ot, epoch_df,
                              initial_epoch="InitVar", 
                              respond_epoch="Crash",
                              equilibrium_epoch="CrashVar",
                              robustness=6,
                              analysis_metrics=analysis_metrics)
    
    if np.isnan(d).any() or np.isnan(e).any():
        print("  For simulation %s" % sim)
        print("  ...failed to compute detection and/or equilibrium.")
        print("  Skipping.")
        continue 
    
    # Plot
    metrics = ["mean_k", "pi", "f_ibd"]
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    
    for ax, metric in zip(axes.flatten(), metrics):
        genetic_trajectory_plot(metric, ot, epoch_df,
                                "steelblue", ax,
                                norm_t0=("Crash", "t0"),
                                indicate_epochs=[("InitVar", "t0"), ("Crash", "t0")],
                                time_limits=epoch_df.loc["InitVar", "t0"] + (0, 100*365),
                                t_detection=d,
                                t_equilibrium=e,
                                years_per_major_tick=10)
        ax.set_ylabel(genetic_names[metric])
        if metric == "f_ibd":
            ax.set_xlabel("Time [years]")
        
    if savefig:
        fig.savefig(os.path.join(example_path, "crash_%s") % sim, 
                    bbox_inches="tight", pad_inches=0.5)
print("Done.")
print("")




# RECOVERY STATISTICS
print("Calculating response times for recovery...")
ds = []
es = []
print("-"*80)
for sim in sim_complete:
    print("%s..." % sim)
    # Load full data frame
    epoch_df = pd.read_csv(os.path.join(expt_path, sim, "epoch_df.csv"))
    epoch_df.index = epoch_df.name
    og = pd.read_csv(os.path.join(expt_path, sim, "og.csv"))
    op = pd.read_csv(os.path.join(expt_path, sim, "op.csv"))

    # Merge on time
    ot = pd.merge(left=op, right=og, on="t0")
    
    # Compute statistics
    d = calc_detection_time(ot, epoch_df, 
                            initial_epoch="CrashVar", 
                            respond_epoch="Recovery", 
                            robustness=3, alpha=0.01,
                            analysis_metrics=analysis_metrics)
    
    e = calc_equilibrium_time(ot, epoch_df,
                              initial_epoch="CrashVar", 
                              respond_epoch="Recovery",
                              equilibrium_epoch="InitVar",
                              robustness=6,
                              analysis_metrics=analysis_metrics)
    
    # Store
    ds.append(d)
    es.append(e)
    
print("-"*80)
print("Aggregating...")
# ds = [d for d in ds if not np.isnan(d).any()]
# es = [e for e in es if not np.isnan(e).any()]
detect_df = pd.concat(ds, 1).transpose()
equil_df = pd.concat(es, 1).transpose()
# Store medians
response_dt["recovery_detect"] = detect_df.median().values
response_dt["recovery_equil"] = equil_df.median().values
print("Saving...")
detect_df.to_csv(os.path.join(output_path, "recovery_detection.csv"), index=False)
equil_df.to_csv(os.path.join(output_path, "recovery_equilibrium.csv"), index=False)
print("Done.")
print("")




# SAVE MEDIANS
print("Saving medians...")
fn = os.path.join(output_path, "response_medians.csv")
response_df = pd.DataFrame(response_dt)
response_df.to_csv(fn)
print("  to: %s" % fn)
print("Done.")
print("")




# VISUALIZE SELECTED FOR RECOVERY
print("Visualizing randomly selected set...")
n_view = 5
view = np.random.choice(len(sim_complete), n_view, replace = False)
print("  #: %d" % n_view)
print("  to: %s" % example_path)

for ix in view:
    
    # Select individual
    sim = sim_complete[ix]
    
    # Load full data frame
    epoch_df = pd.read_csv(os.path.join(expt_path, sim, "epoch_df.csv"))
    epoch_df.index = epoch_df.name
    og = pd.read_csv(os.path.join(expt_path, sim, "og.csv"))
    op = pd.read_csv(os.path.join(expt_path, sim, "op.csv"))

    # Merge on time
    ot = pd.merge(left=op, right=og, on="t0")

    # Compute statistics
    d = calc_detection_time(ot, epoch_df, 
                            initial_epoch="CrashVar", 
                            respond_epoch="Recovery", 
                            robustness=3, alpha=0.01,
                            analysis_metrics=analysis_metrics)

    e = calc_equilibrium_time(ot, epoch_df,
                              initial_epoch="CrashVar", 
                              respond_epoch="Recovery",
                              equilibrium_epoch="InitVar",
                              robustness=6,
                              analysis_metrics=analysis_metrics)
    
    if np.isnan(d).any() or np.isnan(e).any():
        print("  For simulation %s" % sim)
        print("  ...failed to compute detection and/or equilibrium.")
        print("  Skipping.")
        continue 
    
    # Visualize
    metrics = ["mean_k", "pi", "f_ibd"]
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    
    for ax, metric in zip(axes.flatten(), metrics):
        genetic_trajectory_plot(metric, ot, epoch_df,
                                "steelblue", ax,
                                norm_t0=("Recovery", "t0"),
                                indicate_epochs=[("CrashVar", "t0"), ("Recovery", "t0")],
                                time_limits=epoch_df.loc["CrashVar", "t0"] + (0, 100*365),
                                t_detection=d,
                                t_equilibrium=e,
                                years_per_major_tick=10)
        ax.set_ylabel(genetic_names[metric])
        if metric == "f_ibd":
            ax.set_xlabel("Time [years]")
        
    if savefig:
        fig.savefig(os.path.join(example_path, "recovery_%s") % sim, 
                    bbox_inches="tight", pad_inches=0.5)
print("Done.")
print("")




print("-" * 80)
print("Finished at: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 80)
