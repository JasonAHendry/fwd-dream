import os
import sys
from datetime import datetime
import getopt
import configparser
from lib.preferences import *
from lib.intervention import *


print("=" * 80)
print("Analyze the binned & averaged results of a forward-dream experiment.")
print("NB: `bin_intervention_results.py` must be run first.")
print("-" * 80)
print("Command: %s" % " ".join(sys.argv))
print("Run on host: %s" % os.uname().nodename)
print("Operating system: %s" % os.uname().sysname)
print("Machine: %s" % os.uname().machine)
print("Started at: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 80)


# PARSE COMMAND-LINE
try:
    opts, args = getopt.getopt(sys.argv[1:], ":e:p:")
except getopt.GetoptError:
    print("Option Error. Please conform to:")
    print("-e <str> -p <str>")

for opt, value in opts:
    if opt == "-e":
        expt_name = value
        expt_path = os.path.join("results", expt_name)
        bin_path = os.path.join("analysis", expt_name)
        output_path = os.path.join(bin_path, "figs")
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        print("Preparing to analyze", expt_name)
        print(" Experiment Path:", expt_path)
        print(" Binned Data Path:", bin_path)
        print(" Output Path:", output_path)
    elif opt == "-p":
        param_path = value
        param_file = os.path.basename(param_path)
        param_set = param_file.split("_")[1].split(".")[0]
        print(" Parameter path:", param_path)
        print(" Parameter file:", param_file)
        print(" Parameter set:", param_set)
    else:
        print("Parameter %s not recognized." % opt)
        sys.exit(2)


# LOAD SIMULATION EPOCHS AND PARAMETERS
print("Loading parameters and epochs.")
epoch_df = pd.read_csv(os.path.join(bin_path, "epoch_df.csv"))  # assumes epochs are same for all replicates
max_t0 = epoch_df.t1.max()
config = configparser.ConfigParser()
config.read(param_path)
track_ibd = config.getboolean('Sampling', 'track_ibd')
if track_ibd:
    genetic_metrics += ibd_metrics


# LOAD PREVALENCE AND DIVERSITY
print("Loading prevalence and diversity data.")
ops_mean = pd.read_csv(bin_path + "/ops_mean.csv")
ops_std = pd.read_csv(bin_path + "/ops_stds.csv")
ops_se = pd.read_csv(bin_path + "/ops_se.csv")
ogs_mean = pd.read_csv(bin_path + "/ogs_mean.csv")
ogs_std = pd.read_csv(bin_path + "/ogs_stds.csv")
ogs_se = pd.read_csv(bin_path + "/ogs_se.csv")


# INTERVENTION TRAJECTORIES
print("Plotting Intervention Trajectory for each Diversity Metric...")
full_output_path = os.path.join(output_path, "full")
if not os.path.isdir(full_output_path):
    os.mkdir(full_output_path)
epochs_output_path = os.path.join(output_path, "epochs")  # maybe need to remove
if not os.path.isdir(epochs_output_path):
    os.mkdir(epochs_output_path)

prevalence_diversity_plot(op=ops_mean, op_se=ops_se,
                          og=ogs_mean, og_se=ogs_se,
                          epoch_df=epoch_df,
                          genetic_names=genetic_names,
                          genetic_metrics=genetic_metrics,
                          focus_metric=None,
                          make_pdf=True, output_path=full_output_path)

for focus_metric in genetic_metrics:
    print(" Plotting", focus_metric)
    # Full Simulation Duration
    prevalence_diversity_plot(op=ops_mean, op_se=ops_se,
                              og=ogs_mean, og_se=ogs_se,
                              epoch_df=epoch_df,
                              genetic_names=genetic_names,
                              genetic_metrics=genetic_metrics,
                              focus_metric=focus_metric,
                              make_pdf=True, output_path=full_output_path)
    # Only the Epochs
    epochs_diversity_plot(op=ops_mean, op_se=ops_se,
                          og=ogs_mean, og_se=ogs_se,
                          epoch_df=epoch_df,
                          genetic_names=genetic_names,
                          genetic_metrics=genetic_metrics,
                          focus_metric=focus_metric,
                          make_pdf=True, output_path=epochs_output_path)


# EARLIEST DETECTION
print("Performing Detection Time Analysis")
early_path = os.path.join(bin_path, "early")
if not os.path.isdir(early_path):
    os.mkdir(early_path)
early_fig_path = os.path.join(early_path, "figs")
if not os.path.isdir(early_fig_path):
    os.mkdir(early_fig_path)

# Crash -- Detection Time
op_crash_detect = calc_earliest_detect_df(ops_metrics, 
                                          equil_epoch="InitVar", 
                                          adj_epoch="Crash",
                                          epoch_df=epoch_df, 
                                          df_means=ops_mean, 
                                          df_stds=ops_std)
og_crash_detect = calc_earliest_detect_df(genetic_metrics, 
                                          equil_epoch="InitVar", 
                                          adj_epoch="Crash",
                                          epoch_df=epoch_df, 
                                          df_means=ogs_mean, 
                                          df_stds=ogs_std)
crash_detect = pd.concat([op_crash_detect, og_crash_detect])

# Crash -- Equilibrium Time
op_crash_equilibrium = calc_earliest_equilibrium_df(ops_metrics, 
                                                    equil_epoch="CrashVar", 
                                                    adj_epoch="Crash",
                                                    epoch_df=epoch_df, 
                                                    df_means=ops_mean, 
                                                    df_stds=ops_std)
og_crash_equilibrium = calc_earliest_equilibrium_df(genetic_metrics, 
                                                    equil_epoch="CrashVar", 
                                                    adj_epoch="Crash",
                                                    epoch_df=epoch_df, 
                                                    df_means=ogs_mean, 
                                                    df_stds=ogs_std)
crash_equilibrium = pd.concat([op_crash_equilibrium, og_crash_equilibrium])

# Crash -- Merged
crash_dynamics = pd.merge(crash_detect, 
                          crash_equilibrium, 
                          how='inner', on='metric', 
                          suffixes=['_detect', '_equil'])
crash_dynamics["overlap"] = crash_dynamics.apply(lambda row: 
                                                 overlap([row["l_detect"], row["u_detect"]],
                                                         [row["l_equil"], row["u_equil"]]),
                                                 axis=1)

# Recovery -- Detection Time
op_recovery_detect = calc_earliest_detect_df(ops_metrics, 
                                             equil_epoch="CrashVar", 
                                             adj_epoch="Recovery",
                                             epoch_df=epoch_df, 
                                             df_means=ops_mean, 
                                             df_stds=ops_std)
og_recovery_detect = calc_earliest_detect_df(genetic_metrics, 
                                             equil_epoch="CrashVar", 
                                             adj_epoch="Recovery",
                                             epoch_df=epoch_df, 
                                             df_means=ogs_mean, 
                                             df_stds=ogs_std)
recovery_detect = pd.concat([op_recovery_detect, og_recovery_detect])

# Recovery - Equilibrium Time
op_recovery_equilibrium = calc_earliest_equilibrium_df(ops_metrics, 
                                                       equil_epoch="InitVar",
                                                       adj_epoch="Recovery",
                                                       epoch_df=epoch_df, 
                                                       df_means=ops_mean, 
                                                       df_stds=ops_std)
og_recovery_equilibrium = calc_earliest_equilibrium_df(genetic_metrics, 
                                                       equil_epoch="InitVar", 
                                                       adj_epoch="Recovery",
                                                       epoch_df=epoch_df, 
                                                       df_means=ogs_mean, 
                                                       df_stds=ogs_std)
recovery_equilibrium = pd.concat([op_recovery_equilibrium, og_recovery_equilibrium])

# Recovery - Merge
recovery_dynamics = pd.merge(recovery_detect, 
                             recovery_equilibrium, 
                             how='inner', on='metric', 
                             suffixes=['_detect', '_equil'])
recovery_dynamics["overlap"] = recovery_dynamics.apply(lambda row: 
                                                       overlap([row["l_detect"], row["u_detect"]],
                                                               [row["l_equil"], row["u_equil"]]), 
                                                       axis=1)

# Storage
crash_dynamics.to_csv(os.path.join(early_path, "crash_dynamics.csv"), index=False)
recovery_dynamics.to_csv(os.path.join(early_path, "recovery_dynamics.csv"), index=False)

# Plot
# Prepare Colours
genetic_names.update(op_names)  # update genetic_names
genetic_grps.update(op_grps)
n_grps = len(np.unique(list(genetic_grps.values())))
basic_grp_cols = sns.color_palette("Paired", n_grps-1) + [(0.5, 0.5, 0.5)]
col_dt = {k: basic_grp_cols[v-1] for k,v in list(genetic_grps.items())}
# Summaries
plot_response_dynamics(df=crash_dynamics, 
                       detect_name="tdelay_detect", equil_name="tdelay_equil",
                       title="Crash Dynamics \n for %s Intervention" % param_set,
                       genetic_names=genetic_names,
                       genetic_grps=genetic_grps,
                       col_dt=col_dt,
                       cut_days=90, max_years=None,
                       make_png=True, output_path=early_fig_path, png_name="crash.png")

plot_response_dynamics(df=recovery_dynamics, 
                       detect_name="tdelay_detect", equil_name="tdelay_equil",
                       title="Recovery Dynamics \n for %s Intervention" % param_set,
                       genetic_names=genetic_names,
                       genetic_grps=genetic_grps,
                       col_dt=col_dt,
                       cut_days=90, max_years=None,
                       make_png=True, output_path=early_fig_path, png_name="recovery.png")
# Trajectories
for focus_metric in genetic_metrics:
    plot_response_trajectory(focus_metric, focus_epoch="Crash", epoch_df=epoch_df,
                             df=crash_dynamics,
                             track="detect",
                             ops_mean=ops_mean,
                             ogs_mean=ogs_mean,
                             genetic_names=genetic_names,
                             col_dt=col_dt, 
                             make_png=True, output_path=early_fig_path)
    
    plot_response_trajectory(focus_metric, focus_epoch="Recovery", epoch_df=epoch_df,
                             df=recovery_dynamics,
                             track="detect",
                             ops_mean=ops_mean,
                             ogs_mean=ogs_mean,
                             genetic_names=genetic_names,
                             col_dt=col_dt, 
                             make_png=True, output_path=early_fig_path)
    

# LINKAGE DECAY
track_r2 = config.getboolean('Sampling', 'track_r2')
if track_r2:
    print("Plotting Linkage Decay.")
    r2_output_path = os.path.join(output_path, "r2")
    # Load Data
    r2_t0 = np.load(os.path.join(bin_path, "r2_t0.npy"))
    r2_bins = np.load(os.path.join(bin_path, "r2_bins.npy"))
    r2_mean = np.load(os.path.join(bin_path, "r2_mean.npy"))
    r2_se = np.load(os.path.join(bin_path, "r2_se.npy"))
    # Define Output Dirv
    if not os.path.isdir(r2_output_path):
        os.mkdir(r2_output_path)
    # Plot
    r2_by_epoch_plot(r2_t0=r2_t0, r2_mean=r2_mean, r2_se=r2_se, r2_bins=r2_bins, epoch_df=epoch_df,
                     make_pdf=True, output_path=r2_output_path)
    r2_through_time_plot(r2_t0=r2_t0, r2_mean=r2_mean, r2_se=r2_se, r2_bins=r2_bins, epoch_df=epoch_df,
                         make_pdf=True, output_path=r2_output_path)
    r2_epochs_time_only(r2_t0=r2_t0, r2_mean=r2_mean, r2_se=r2_se, r2_bins=r2_bins, epoch_df=epoch_df,
                        make_pdf=True, output_path=r2_output_path)
    print("Done.")


# SITE-FREQUENCY
track_sfs = config.getboolean('Sampling', 'track_sfs')
if track_sfs:
    print("Plotting SFS.")
    sfs_output_path = os.path.join(output_path, "sfs")
    # Load Data
    sfs_t0 = np.load(os.path.join(bin_path, "sfs_t0.npy"))
    sfs_bins = np.load(os.path.join(bin_path, "sfs_bins.npy"))
    sfs_mean = np.load(os.path.join(bin_path, "sfs_mean.npy"))
    sfs_se = np.load(os.path.join(bin_path, "sfs_se.npy"))
    # Define Output Dir
    if not os.path.isdir(sfs_output_path):
        os.mkdir(sfs_output_path)
    # Plot
    sfs_by_epoch_plot(sfs_t0=sfs_t0, sfs_mean=sfs_mean, sfs_se=sfs_se, sfs_bins=sfs_bins, epoch_df=epoch_df,
                      make_pdf=True, output_path=sfs_output_path)
    sfs_through_time_plot(sfs_t0=sfs_t0, sfs_mean=sfs_mean, sfs_se=sfs_se, sfs_bins=sfs_bins, epoch_df=epoch_df,
                          make_pdf=True, output_path=sfs_output_path)
    sfs_epochs_time_only(sfs_t0=sfs_t0, sfs_mean=sfs_mean, sfs_se=sfs_se, sfs_bins=sfs_bins, epoch_df=epoch_df,
                         make_pdf=True, output_path=sfs_output_path)
    print("Done.")
    
    
print("-" * 80)
print("Finished at: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 80)

