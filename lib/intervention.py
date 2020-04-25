import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
import seaborn as sns

sns.set_style("white")


# ================================================================= #
# Binning Functions
#
# ================================================================= #


def load_simulations(dirs, file_name):
    """
    Load a given file from all simulations within
    an experiment
    """
    dfs = []
    for d in dirs:
        df = pd.read_csv(os.path.join(d, file_name))
        dfs.append(df)
    return dfs


def load_simulations_from_npy(dirs, file_name, columns, times):
    """
    Load a given file from all simulations within
    an experiment
    """
    dfs = []
    for d in dirs:
        df = pd.DataFrame(np.load(os.path.join(d, file_name)),
                          columns=columns)
        df['t0'] = np.load(os.path.join(d, times))
        dfs.append(df)
    return dfs


def bin_simulations(simulations, keep_cols, bins, right=True):
    """
    Bin a set of simulations
    """
    sims_binned = []
    for sim in simulations:
        grp = sim.groupby(pd.cut(sim.t0, bins=bins, right=right))
        sim_binned = grp.mean()[keep_cols]
        sims_binned.append(sim_binned)
    return sims_binned


def average_simulations(simulations_binned, keep_cols, bin_midpoints):
    """
    Average a set of binned simulations
    """
    sims_array = np.dstack(simulations_binned)
    print(" Bins:", sims_array.shape[0])
    print(" Metrics:", sims_array.shape[1])
    print(" Experiments:", sims_array.shape[2])

    # Count number of simulations with data for each bin
    nan_array = np.isnan(sims_array)
    sims_counts = np.sum(~nan_array, 2)  # rows: bins, cols: metrics, values: # expts not NA

    # Remove bins with no data
    sims_array = sims_array[sims_counts.sum(1) != 0]  # remove bins with zero entries
    bin_midpoints = bin_midpoints[sims_counts.sum(1) != 0]
    sims_counts = sims_counts[sims_counts.sum(1) != 0]

    # Compute statistics, cognizant of nans
    sims_means = np.nanmean(sims_array, 2)
    sims_stds = np.nanstd(sims_array, 2)
    sims_se = sims_stds / np.sqrt(sims_counts)

    # Pandas dfs; bin midpoints become new times
    sims_means = pd.DataFrame(sims_means, columns=keep_cols)
    sims_means["t0"] = bin_midpoints
    sims_stds = pd.DataFrame(sims_stds, columns=keep_cols)
    sims_stds["t0"] = bin_midpoints
    sims_se = pd.DataFrame(sims_se, columns=keep_cols)
    sims_se["t0"] = bin_midpoints
    return sims_array, sims_means, sims_stds, sims_se


# ================================================================= #
# Plotting Functions
#
# ================================================================= #

# ---------------------------------
# Prevalence and Summary Statistics
#
# ---------------------------------


def prevalence_diversity_plot(op, op_se, og, og_se, epoch_df,
                              genetic_alpha=0.08,
                              n_se=1.96, se_alpha=0.2,
                              genetic_names=None,
                              genetic_metrics=None,
                              focus_metric=None,
                              time_limits=None,
                              make_pdf=False, output_path=None):
    """
    Plot Prevalence & Genetic Diversity Metrics
    of a Single Simulation

    """
    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    # Host Prevalence
    ax.plot(op["t0"], op["HX"], color="blue", linewidth=0.8, label="Host All")
    ax.fill_between(x=op["t0"],
                    y1=op["HX"] - op_se["HX"] * n_se,
                    y2=op["HX"] + op_se["HX"] * n_se,
                    color="blue", alpha=se_alpha)
    ax.plot(op["t0"], op["HmX"], color="darkblue", linewidth=0.8, label="Host Mixed")
    ax.fill_between(x=op["t0"],
                    y1=op["HmX"] - op_se["HmX"] * n_se,
                    y2=op["HmX"] + op_se["HmX"] * n_se,
                    color="darkblue", alpha=se_alpha)
    # Vector Prevalence
    ax.plot(op["t0"], op["VX"], color="red", linewidth=0.8, label="Vector All")
    ax.fill_between(x=op["t0"],
                    y1=op["VX"] - op_se["VX"] * n_se,
                    y2=op["VX"] + op_se["VX"] * n_se,
                    color="red", alpha=se_alpha)
    ax.plot(op["t0"], op["VmX"], color="darkred", linewidth=0.8, label="Vector Mixed")
    ax.fill_between(x=op["t0"],
                    y1=op["VmX"] - op_se["VmX"] * n_se,
                    y2=op["VmX"] + op_se["VmX"] * n_se,
                    color="darkred", alpha=se_alpha)
    # Delineate Epochs and Equilibriums
    for i, row in epoch_df.iterrows():
        ax.vlines(ymin=0, ymax=1, x=row['t0'], color="darkgrey")
        ax.hlines(xmin=row['t0'], xmax=row['t1'], y=row['x_h'], color="blue", linestyle="--")
        ax.hlines(xmin=row['t0'], xmax=row['t1'], y=row['x_v'], color="red", linestyle="--")
    # Limits
    ax.set_ylim([0, 1])
    if time_limits is not None:
        ax.set_xlim(time_limits)
    else:
        epoch_t0 = epoch_df.iloc[0].t0
        epoch_t1 = epoch_df.iloc[-1].t1
        time_limits = [epoch_t0, epoch_t1]
        ax.set_xlim(time_limits)
    # Ticks
    days_per_year = 365
    years_per_major_tick = 10

    def format_xaxis(value, tick_number,
                     days_per_year=days_per_year,
                     years_per_major_tick=years_per_major_tick):
        """
        Plot as Decade Number as opposed to Day Number
        """
        return int(round(value / (days_per_year * years_per_major_tick)))

    ax.tick_params(axis='both', which='major', direction='in', length=10, labelsize=16)
    ax.tick_params(axis='both', which='minor', direction='in', length=8, labelsize=16)
    ax.xaxis.set_major_locator(plt.MultipleLocator(days_per_year * years_per_major_tick))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(days_per_year))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_xaxis))
    # Text
    ax.set_title("Simulating Bednet Intervetion", fontsize=20)
    ax.set_xlabel("Time (decades)", fontsize=18)
    ax.set_ylabel("Prevalence", fontsize=18)
    # Legend
    leg = ax.legend(ncol=2, fontsize=16)
    for l in leg.legendHandles:
        l.set_linewidth(3.0)
    # Plot Genetic Diversity Lines
    if genetic_metrics is not None:
        for metric in genetic_metrics:
            if metric is focus_metric:
                axm = ax.twinx()
                axm.plot(og["t0"], og[metric], "orange")
                axm.fill_between(x=og["t0"],
                                 y1=og[metric] - og_se[metric] * n_se,
                                 y2=og[metric] + og_se[metric] * n_se,
                                 color="orange", alpha=se_alpha)
                axm.set_ylabel(genetic_names[metric], fontsize=18)
                axm.tick_params(axis='y', which='major', direction='in', length=10, labelsize=16)
                axm.set_xlim(time_limits)
            else:
                axm = ax.twinx()
                axm.plot(og["t0"], og[metric], "forestgreen", alpha=genetic_alpha)
                axm.axes.get_yaxis().set_visible(False)
                axm.set_xlim(time_limits)
    # Output
    if make_pdf and output_path is not None:
        fig.savefig(os.path.join(output_path, str(focus_metric) + ".png"))
        plt.close(fig)


def epochs_diversity_plot(**kwargs):
    """
    Plot Prevalence & Diversity Metrics,
    of all epochs
    """
    time_min = kwargs["epoch_df"]["t1"][0]  # end of init
    time_max = kwargs["epoch_df"]["t1"][len(kwargs["epoch_df"]) - 1]
    time_limits = [time_min, time_max]
    prevalence_diversity_plot(time_limits=time_limits, **kwargs)


def epoch_diversity_plot(epoch_indx, **kwargs):
    """
    Plot Prevalence & Diversity Metrics,
    of a single epoch
    """
    time_min = kwargs["epoch_df"]["t0"][epoch_indx]
    time_max = kwargs["epoch_df"]["t1"][epoch_indx]
    time_limits = [time_min, time_max]
    prevalence_diversity_plot(time_limits=time_limits, **kwargs)


# ---------------------------------
# SFS
#
# ---------------------------------


def sfs_by_epoch_plot(sfs_t0, sfs_mean, sfs_se, sfs_bins, epoch_df,
                      n_se=1.96, se_alpha=0.2, pal='viridis',
                      make_pdf=False, output_path=None):
    """
    Plot the SFS at the end of each Epoch
    """
    n_epochs = len(epoch_df)
    sns.set_style("ticks")
    fig, ax = plt.subplots(1, n_epochs, figsize=(n_epochs * 5, 4), sharey=True)
    fig.subplots_adjust(wspace=0.1, bottom=0.2)

    for i, e in epoch_df.iterrows():

        end_time = e.t1
        closest_time = sfs_t0[(end_time - sfs_t0) > 0][-1]
        indx = np.where(sfs_t0 == closest_time)[0][0]

        col_pal = sns.color_palette(pal, len(sfs_bins))
        for j in np.arange(len(sfs_bins)):
            ax[i].plot(sfs_bins[j],
                       sfs_mean[indx, j],
                       marker='o', color=col_pal[j])
            ax[i].plot(sfs_bins[j:j + 2],
                       sfs_mean[indx, j:j + 2],
                       linewidth=2, color=col_pal[j])
            ax[i].fill_between(x=sfs_bins[j:j + 2],
                               y1=sfs_mean[indx, j:j + 2] - sfs_se[indx, j:j + 2] * n_se,
                               y2=sfs_mean[indx, j:j + 2] + sfs_se[indx, j:j + 2] * n_se,
                               alpha=se_alpha, color=col_pal[j])
        # Ticks
        ax[i].set_xticks(sfs_bins + 0.05)
        xticklabs = ["%.01f - %.01f" % (s - 0.05, s + 0.05) for s in sfs_bins]
        ax[i].set_xticklabels(xticklabs, rotation=-45, ha='center', va='baseline')
        ax[i].grid(linestyle='--')
        # Text
        ax[i].set_ylabel("$ \zeta_{freq.} $", fontsize=14)
        ax[i].set_xlabel("Site Frequency", fontsize=14)
        ax[i].set_title(e["name"], fontsize=16)
        # Output
        if make_pdf and output_path is not None:
            fig.savefig(os.path.join(output_path, "sfs_by_epoch.png"))
            plt.close(fig)


def sfs_through_time_plot(sfs_t0, sfs_mean, sfs_se, sfs_bins, epoch_df,
                          time_limits=None,
                          n_se=1.96, se_alpha=0.2, pal='viridis',
                          make_pdf=False, output_path=None):
    """
    Plot the Site Frequency Spectrum as it evolves through
    time
    """
    sns.set_palette(pal, len(sfs_bins))
    fig, ax = plt.subplots(1, 1, figsize=(20, 6))

    sfs_bin_size = np.diff(sfs_bins).max()

    for i, sfs_bin in enumerate(sfs_bins):
        ax.plot(sfs_t0, sfs_mean[:, i], label="%.01f - %.01f" % (sfs_bin - sfs_bin_size/2.0, sfs_bin + sfs_bin_size/2.0))
        ax.fill_between(x=sfs_t0,
                        y1=sfs_mean[:, i] - sfs_se[:, i] * n_se,
                        y2=sfs_mean[:, i] + sfs_se[:, i] * n_se,
                        alpha=se_alpha)
    # Epoch Lines
    ax.set_autoscaley_on(False)
    ymin = -10
    ymax = sfs_mean.max() + sfs_se.max() * n_se * 2 # this will definitely cover
    for i, row in epoch_df.iterrows():
        ax.vlines(ymin=ymin, ymax=ymax, x=row['t0'], color="darkgrey")
    # Limits
    if time_limits is not None:
        ax.set_xlim(time_limits)
    # Ticks
    days_per_year = 365
    years_per_major_tick = 10

    def format_xaxis(value, tick_number, d=days_per_year, m=years_per_major_tick):
        """
        Plot as Decade Number as opposed to Day Number
        """
        return int(round(value / (d * m)))

    ax.tick_params(axis='both', which='major', direction='in', length=10, labelsize=16)
    ax.tick_params(axis='both', which='minor', direction='in', length=8, labelsize=16)
    ax.xaxis.set_major_locator(plt.MultipleLocator(days_per_year * years_per_major_tick))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_xaxis))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(days_per_year))
    # Text
    ax.set_title("Simulating Bednet Intervetion \n Site-Frequency Spectrum", fontsize=20)
    ax.set_xlabel("Time (decades)", fontsize=18)
    ax.set_ylabel("$ \zeta_{freq.} $", fontsize=18)
    # Legend
    l = ax.legend(fontsize=14, ncol=2, frameon=True, title="Frequency")
    l = plt.setp(l.get_title(), fontsize=16)
    # Output
    if make_pdf and output_path is not None:
        if time_limits is not None:
            fig.savefig(os.path.join(output_path, "sfs_epochs_time.png"))
        else:
            fig.savefig(os.path.join(output_path, "sfs_through_time.png"))
        plt.close(fig)


def sfs_epochs_time_only(**kwargs):
    """
    Plot the SFS through time, but focusing only
    on the Epochs
    """
    time_min = kwargs["epoch_df"]["t1"][0]  # end of init
    time_max = kwargs["epoch_df"]["t1"][len(kwargs["epoch_df"]) - 1]
    time_limits = [time_min, time_max]
    sfs_through_time_plot(time_limits=time_limits, **kwargs)


# ---------------------------------
# r2
#
# ---------------------------------


def r2_by_epoch_plot(r2_t0, r2_mean, r2_se, r2_bins, epoch_df,
                     n_se=1.96, se_alpha=0.2, pal='plasma',
                     make_pdf=False, output_path=None):
    """
    Plot Linkage Disequilibrium decay (r2) at the end
    of each Epoch
    """
    n_epochs = len(epoch_df)
    sns.set_style("ticks")
    fig, ax = plt.subplots(1, n_epochs, figsize=(n_epochs * 5, 4), sharey=True)
    fig.subplots_adjust(wspace=0.1, bottom=0.2)

    for i, e in epoch_df.iterrows():

        end_time = e.t1
        closest_time = r2_t0[(end_time - r2_t0) > 0][-1]
        indx = np.where(r2_t0 == closest_time)[0][0]

        col_pal = sns.color_palette(pal, len(r2_bins))
        for j in np.arange(len(r2_bins)):
            ax[i].plot(r2_bins[j],
                       r2_mean[indx, j],
                       marker='o', color=col_pal[j])
            ax[i].plot(r2_bins[j:j + 2],
                       r2_mean[indx, j:j + 2],
                       linewidth=2, color=col_pal[j])
            ax[i].fill_between(x=r2_bins[j:j + 2],
                               y1=r2_mean[indx, j:j + 2] - r2_se[indx, j:j + 2] * n_se,
                               y2=r2_mean[indx, j:j + 2] + r2_se[indx, j:j + 2] * n_se,
                               alpha=se_alpha, color=col_pal[j])
        # Ticks
        ax[i].set_xticks(r2_bins + 2.5)
        r2_bin_size = np.diff(r2_bins).max()
        xtick_labs = ["%.0f - %.0f" % (r - r2_bin_size / 2.0, r + r2_bin_size / 2.0) for r in r2_bins]
        ax[i].set_xticklabels(xtick_labs, rotation=-45, ha="center", va="baseline")
        ax[i].grid(linestyle='--')
        # Text
        ax[i].set_ylabel("$ r^2 $", fontsize=14)
        ax[i].set_xlabel("Distance Between SNPs (bp)", fontsize=14)
        ax[i].set_title(e["name"], fontsize=14)
        # Output
        if make_pdf and output_path is not None:
            fig.savefig(os.path.join(output_path, "r2_by_epoch.png"))
            plt.close(fig)


def r2_through_time_plot(r2_t0, r2_mean, r2_se, r2_bins, epoch_df,
                         time_limits=None, n_se=1.96, se_alpha=0.2, pal="plasma",
                         make_pdf=False, output_path=None):
    """
    Plot Linkage Diseqiulibrium Decay (r2) as it evolves through
    time
    """
    sns.set_palette(pal, len(r2_bins))
    fig, ax = plt.subplots(1, 1, figsize=(20, 6))

    r2_bin_size = np.diff(r2_bins).max()
    for i, r2_bin in enumerate(r2_bins):
        ax.plot(r2_t0, r2_mean[:, i], alpha=0.75,
                label="%.0f - %.0f" % (r2_bin - r2_bin_size / 2.0, r2_bin + r2_bin_size / 2.0))
        ax.fill_between(x=r2_t0,
                        y1=r2_mean[:, i] - r2_se[:, i] * n_se,
                        y2=r2_mean[:, i] + r2_se[:, i] * n_se,
                        alpha=se_alpha)
    # Epoch Lines
    ax.set_autoscaley_on(False)
    for i, row in epoch_df.iterrows():
        ax.vlines(ymin=-50, ymax=50, x=row['t0'], color="darkgrey")
    # Limits
    ax.set_ylim([0, 1])
    if time_limits is not None:
        ax.set_xlim(time_limits)
    # Ticks
    days_per_year = 365
    years_per_major_tick = 10

    def format_xaxis(value, tick_number,
                     days_per_year=days_per_year,
                     years_per_major_tick=years_per_major_tick):
        """
        Plot as Decade Number as opposed to Day Number
        """
        return int(round(value / (days_per_year * years_per_major_tick)))

    ax.tick_params(axis='both', which='major', direction='in', length=10, labelsize=16)
    ax.tick_params(axis='both', which='minor', direction='in', length=8, labelsize=16)
    ax.xaxis.set_major_locator(plt.MultipleLocator(days_per_year * years_per_major_tick))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_xaxis))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(days_per_year))
    # Text
    ax.set_title("Simulating Bednet Intervetion \n Linkage-Disequilibrium Decay", fontsize=20)
    ax.set_xlabel("Time (decades)", fontsize=18)
    ax.set_ylabel("$ r^2 $", fontsize=18)
    # Legend
    l = ax.legend(fontsize=14, ncol=4, frameon=True, title="Distance Between SNPs")
    l = plt.setp(l.get_title(), fontsize=16)
    # Output
    if make_pdf and output_path is not None:
        if time_limits is not None:
            fig.savefig(os.path.join(output_path, "r2_epochs_time.png"))
        else:
            fig.savefig(os.path.join(output_path, "r2_through_time.png"))
        plt.close(fig)


def r2_epochs_time_only(**kwargs):
    """
    Plot the SFS through time, but focusing only
    on the Epochs
    """
    time_min = kwargs["epoch_df"]["t1"][0]  # end of init
    time_max = kwargs["epoch_df"]["t1"][len(kwargs["epoch_df"]) - 1]
    time_limits = [time_min, time_max]
    r2_through_time_plot(time_limits=time_limits, **kwargs)


# ---------------------------------
# Earliest ...
#
# ---------------------------------


def subset_to_epoch(df, epoch_df, epoch_name):
    """
    Subset a `df` to observations that occur
    within a given Epoch

    Parameters
        df: pd.DataFrame (n_obs, metrics)
            Dataframe of observations. Must contain
            a column `t0` specifying the time at which
            each observation was made.
        epoch_df: pd.DataFrame (n_epochs, 7)
            Dataframe of epochs.
        epoch_name: str
            Name of the epoch of interesting
    Returns
        df: pd.DataFrame
            Input `df` subsetted to only contain
            observations collected within
            `epoch_name`
    """
    epoch_t0 = float(epoch_df[epoch_df.name == epoch_name].t0)
    epoch_t1 = float(epoch_df[epoch_df.name == epoch_name].t1)
    return df.query("@epoch_t0 <= t0 <= @epoch_t1")


def overlap(a, b):
    return (a[0] <= b[0] <= a[1]) or (a[0] <= b[1] <= a[1])


def calc_earliest_detection(metric, df_means, df_stds,
                            epoch_df, adj_epoch, equil_epoch,
                            t_detect=1.96):
    """
    Calculate the earliest time in the `adj_epoch`
    at which `metric` would indicate the prevalence has
    deviated from the `equil_epoch` value
    
    """
    eq_means = subset_to_epoch(df_means, epoch_df, equil_epoch)[metric]
    eq_stds = subset_to_epoch(df_stds, epoch_df, equil_epoch)[metric]
    
    eq_mean = np.mean(eq_means[np.isfinite(eq_means)])
    eq_std = np.mean(eq_stds[np.isfinite(eq_stds)])
    
    u = eq_mean + t_detect*eq_std
    l = eq_mean - t_detect*eq_std
    
    epoch_df.index = epoch_df.name
    start_t0 = epoch_df.loc[adj_epoch].t0
    
    adj_df = subset_to_epoch(df_means, epoch_df, adj_epoch)
    adj_means = np.array(adj_df[metric])
    adj_t0s = np.array(adj_df["t0"])
    
    adj_finite = np.isfinite(adj_means)
    adj_means = adj_means[adj_finite]
    adj_t0s = adj_t0s[adj_finite]
    
    adj_bool = (adj_means < l) | (adj_means > u)
    if adj_bool.any():
        detect_ix = np.where(adj_bool)[0].min()
        detect_tdelay = adj_t0s[detect_ix] - start_t0
        detect_direction = adj_means[detect_ix] > u  # True if increasing
    else:
        detect_ix = np.nan
        detect_tdelay = np.nan
        detect_direction = np.nan
    
    epoch_df.reset_index(drop=True, inplace=True)
    
    return [detect_ix, detect_tdelay, detect_direction, l, eq_mean, u, eq_std]


def calc_earliest_equilibrium(metric, df_means, df_stds,
                              epoch_df, adj_epoch, equil_epoch,
                              t_equil=0.1):
    """
    Calculate the earliest time in the `adj_epoch`
    at which `metric` would indicate the prevalence has
    deviated from the `equil_epoch` value
    
    """
    eq_means = subset_to_epoch(df_means, epoch_df, equil_epoch)[metric]
    eq_stds = subset_to_epoch(df_stds, epoch_df, equil_epoch)[metric]
    
    eq_mean = np.mean(eq_means[np.isfinite(eq_means)])
    eq_std = np.mean(eq_stds[np.isfinite(eq_stds)])
    
    u = eq_mean + t_equil*eq_std
    l = eq_mean - t_equil*eq_std
    
    epoch_df.index = epoch_df.name
    start_t0 = epoch_df.loc[adj_epoch].t0
    
    adj_df = subset_to_epoch(df_means, epoch_df, adj_epoch)
    adj_means = np.array(adj_df[metric])
    adj_t0s = np.array(adj_df["t0"])
    
    adj_finite = np.isfinite(adj_means)
    adj_means = adj_means[adj_finite]
    adj_t0s = adj_t0s[adj_finite]
    
    adj_bool = (l < adj_means) & (adj_means < u)
    if adj_bool.any():
        detect_ix = np.where(adj_bool)[0].min()
        detect_tdelay = adj_t0s[detect_ix] - start_t0
        detect_direction = adj_means[detect_ix] > u  # True if increasing
    else:
        detect_ix = np.nan
        detect_tdelay = np.nan
        detect_direction = np.nan
    
    epoch_df.reset_index(drop=True, inplace=True)
    
    return [detect_ix, detect_tdelay, detect_direction, l, eq_mean, u, eq_std]


def calc_earliest_detect_df(genetic_metrics, equil_epoch, adj_epoch, epoch_df,
                            df_means, df_stds):
    """
    Assemble a dataframe containing the earliest detection times
    for a given `equil_epoch` and `adj_epoch`
    """
    
    ll = []
    for metric in genetic_metrics:
        l = calc_earliest_detection(metric=metric,
                                    adj_epoch=adj_epoch,
                                    equil_epoch=equil_epoch,
                                    epoch_df=epoch_df,
                                    df_means=df_means,
                                    df_stds=df_stds,
                                    t_detect=1.96)
        ll.append(l)
    
    columns = ["ix", "tdelay", "direction", "l", "mu", "u", "std"]
    df = pd.DataFrame(ll, columns=columns)
    df.insert(0, "metric", genetic_metrics)
    df.sort_values("tdelay", inplace=True)
    return df


def calc_earliest_equilibrium_df(genetic_metrics, equil_epoch, adj_epoch, epoch_df,
                                 df_means, df_stds):
    """
    Assemble a dataframe containing the earliest detection times
    for a given `equil_epoch` and `adj_epoch`
    """
    
    ll = []
    for metric in genetic_metrics:
        l = calc_earliest_equilibrium(metric=metric,
                                      adj_epoch=adj_epoch,
                                      equil_epoch=equil_epoch,
                                      epoch_df=epoch_df,
                                      df_means=df_means,
                                      df_stds=df_stds,
                                      t_equil=0.1)
        ll.append(l)
    
    columns = ["ix", "tdelay", "direction", "l", "mu", "u", "std"]
    df = pd.DataFrame(ll, columns=columns)
    df.insert(0, "metric", genetic_metrics)
    df.sort_values("tdelay", inplace=True)
    return df


# ================================
# Earliest Plotting




def plot_response_dynamics(df, detect_name, equil_name,
                           title,
                           genetic_names,
                           genetic_grps,
                           col_dt,
                           cut_days=90, max_years=None,
                           make_png=False, output_path=None, png_name=None):
    """
    Horizontal barplot of the detection and equilibrium times
    for all metrics stored in `df`
    
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 8))
    
    divider = make_axes_locatable(ax)
    lax = divider.append_axes("left", size="20%", pad=0.05)
    
    # LEFT PLOT
    # =========
    lax.barh(# Data
            bottom=np.arange(len(df)),
            left=df[detect_name],
            width=df[equil_name]-df[detect_name],
            # Labels
            tick_label=[genetic_names[m] for m in df["metric"]],
            # Aesthetics
            color=[col_dt[m] + (0.2,) if o else col_dt[m]
                   for o, m in zip(df["overlap"], df["metric"])],
            edgecolor='black', linewidth=0.5)
    # Patch delineating Intervention Window
    lax.add_patch(patches.Rectangle(xy=(0, lax.get_ylim()[0]), 
                                    width=30, 
                                    height=lax.get_ylim()[1]-lax.get_ylim()[0], 
                                    color='lightgrey', alpha=0.75, zorder=0)) 

    # Limits
    lax.set_xlim(0, cut_days)
    # Invert
    lax.invert_yaxis()
    # Ticks
    lax.xaxis.set_major_locator(plt.MultipleLocator(90))
    lax.xaxis.set_minor_locator(plt.MultipleLocator(30))
    lax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, tick: int(val/30.)))
    lax.tick_params(axis='x', which='major', direction='out', length=10)
    lax.tick_params(axis='x', which='minor', direction='out', length=5, labelsize=8, gridOn=True)
    # Grid
    lax.grid(axis='x', which='major', linestyle='-', linewidth=1.25)
    lax.grid(axis='x', which='minor', linestyle='dotted', linewidth=0.8, alpha=0.5)
    # Labels
    lax.set_xlabel("Time (Months)")


    # RIGHT PLOT
    # ==========
    ax.barh(# Data
            bottom=np.arange(len(df)),
            left=df[detect_name],
            width=df[equil_name]-df[detect_name],
            # Aesthetics
            color=[col_dt[m] + (0.2,) if o else col_dt[m]
                   for o, m in zip(df["overlap"], df["metric"])],
            edgecolor='black', linewidth=0.5)
    # Invert
    ax.invert_yaxis()
    # Limits
    if max_years is not None:
        ax.set_xlim(cut_days, max_years*360)
    else:
        ax.set_xlim(cut_days, ax.get_xlim()[1])
    ax.set_ylim(lax.get_ylim())
    # Ticks
    ax.xaxis.set_major_locator(plt.MultipleLocator(360*10))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(360))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, tick: int(val/360.)))
    ax.xaxis.set_minor_formatter(plt.FuncFormatter(lambda val, tick: int(val/360.) if 0 < tick <= 9 else ""))
    ax.tick_params(axis='x', which='major', direction='out', length=10)
    ax.tick_params(axis='x', which='minor', direction='out', length=5, labelsize=8, gridOn=True)
    # Grid
    ax.grid(axis='x', which='major', linestyle='-', linewidth=1.25)
    ax.grid(axis='x', which='minor', linestyle='dotted', linewidth=0.8, alpha=0.5)
    # Labels
    ax.set_xlabel("Time (Years)")
    ax.set_title(title, fontsize=14)
    
    if make_png and output_path is not None:
        fig.savefig(os.path.join(output_path, png_name), 
                    bbox_inches="tight", pad_inches=0.5)
        plt.close(fig)


def plot_response_trajectory(metric, focus_epoch, epoch_df,
                             track,
                             df, ops_mean, ogs_mean,
                             genetic_names, col_dt,
                             buffer_years=5, make_png=False,
                             output_path=None):
    """
    Plot the trajectory of a given genetic diversity `metric`
    focusing on the response to intervention
    
    """
    
    # Prepare Data
    epoch_df.index = epoch_df.name
    start_t0 = epoch_df.loc[focus_epoch].t0
    df.index = df.metric
    df = df.loc[metric]
    tdelay_col = "tdelay_" + track
    tdelay = df[tdelay_col]
    found = True
    if not np.isfinite(tdelay):
        found = False
        tdelay = epoch_df.loc[focus_epoch].t1
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    # Prevalence
    ax.plot(ops_mean["t0"], ops_mean["HX"], color="blue")
    ax.plot(ops_mean["t0"], ops_mean["VX"], color="red")
    ax.plot(ops_mean["t0"], ops_mean["HmX"], color="darkblue")
    ax.plot(ops_mean["t0"], ops_mean["VmX"], color="darkred")
    # Metric
    tax = ax.twinx()
    tax.plot(ogs_mean["t0"], ogs_mean[metric], color="darkgrey", linewidth=3)
    # Limits
    ax.set_xlim(start_t0 - buffer_years*360, start_t0 + tdelay + buffer_years*360)
    ax.set_ylim(0, 1.0)
    # Lines
    ax.axvline(start_t0, color="black")
    if found:
        ax.axvline(start_t0 + tdelay, color="magenta")
    for col in ["u", "mu", "l"]:
        col_name = col + "_" + track
        tax.axhline(df[col_name],
                    color="darkgrey",
                    linestyle="dashed" if col == "mu" else "dotted", zorder=-1)
    # Ticks
    if tdelay >  (360*20):
        ax.xaxis.set_major_locator(plt.MultipleLocator(360*10))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(360))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, t: int(v/(360.0*10))))
        ax.set_xlabel("Time (Decades)")
    else:
        ax.xaxis.set_major_locator(plt.MultipleLocator(360))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(30))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, t: int(v/(360.0)))) 
        ax.set_xlabel("Time (Years)")
    ax.tick_params(axis="both", which="major", direction="out", length=8)
    tax.tick_params(axis="y", which="major", direction="out", length=8)
    ax.tick_params(axis="both", which="minor", direction="out", length=4)

    # Labels
    ax.set_ylabel("Prevalence")
    tax.set_ylabel(genetic_names[metric])
    
    df.reset_index(drop=True, inplace=True)
    epoch_df.reset_index(drop=True, inplace=True)
    
    if make_png and output_path is not None:
        fig.savefig(os.path.join(output_path, focus_epoch.lower() + "-" + metric + ".png"),
                    bbox_inches="tight", pad_inches=0.5)
        plt.close(fig)


        

# ---------------------------------
# Animation Classes
#
# ---------------------------------




class AnimatePrevalence(object):
    """
    Use this class to hold all of the prevalence
    trajectories for a given set of simulations
    TODO:
    - add the fill_between; shouldn't be *too* hard.
    """

    def __init__(self, ax, ops_means, ops_se, epoch_df):

        self.ax = ax

        self.ops_means = ops_means
        self.ops_se = ops_se
        self.epoch_df = epoch_df

        self.max_t0 = self.epoch_df.t1.max()

        self.n_lines = 0
        self.lines = []
        self.line_names = []
        self.line_labels = []
        self.line_colors = []

    def initialize(self, title, line_names, colors, labels):
        """
        lines is a list of lines you want to plot
        """

        self.n_lines = len(line_names)
        self.line_names = line_names
        self.line_colors = colors
        self.line_labels = labels

        # Text
        self.ax.tick_params(axis='both', which='major', labelsize=11)
        self.ax.set_title(title, fontsize=14)
        self.ax.set_xlabel("Time (decades)", fontsize=12)
        self.ax.set_ylabel("Prevalence", fontsize=12)

        # Limits; eventually want to allow for flexibility here
        self.ax.set_xlim([0, self.max_t0])
        self.ax.set_ylim([0, 1])

        # Ticks
        days_per_year = 365
        years_per_major_tick = 10

        def format_xaxis(value, tick_number, d=days_per_year, m=years_per_major_tick):
            """
            Plot as Decade Number as opposed to Day Number
            """
            return int(round(value / (d * m)))

        self.ax.tick_params(axis='both', which='major', direction='in', length=10)
        self.ax.tick_params(axis='both', which='minor', direction='in', length=8)
        self.ax.xaxis.set_major_locator(plt.MultipleLocator(days_per_year * years_per_major_tick))
        self.ax.xaxis.set_major_formatter(plt.FuncFormatter(format_xaxis))
        self.ax.xaxis.set_minor_locator(plt.MultipleLocator(days_per_year))
        self.ax.grid(linestyle="--")

        # Delineate Epochs and Equilibriums
        for i, row in self.epoch_df.iterrows():
            self.ax.vlines(ymin=0, ymax=1, x=row['t0'], color="darkgrey")
            self.ax.hlines(xmin=row['t0'], xmax=row['t1'], y=row['x_h'], color="blue", linestyle="--")
            self.ax.hlines(xmin=row['t0'], xmax=row['t1'], y=row['x_v'], color="red", linestyle="--")

        # Initialize Lines
        for color, label in zip(colors, labels):
            line, = self.ax.plot([], [], linewidth=0.8, color=color, label=label)
            self.lines.append(line)

    def update_line(self, j, line_name):
        indx = [i for i in range(self.n_lines) if line_name == self.line_names[i]][0]
        x = self.ops_means["t0"][:j]
        y = self.ops_means[line_name][:j]
        self.lines[indx].set_data(x, y)

    def update_lines(self, j):
        for line_name in self.line_names:
            self.update_line(j, line_name=line_name)


class AnimateDensity(object):
    """
    Use this class to hold a single
    density plot animation
    Qs
    - How to determine the best way to bin the histogram,
    and specify it's ylimits?
    - The n_expertiments/n_bins ~ # experiments per bin
    - Could determine max count before
    - Anyways, these are minor details
    """

    def __init__(self, ax, metric, metric_indxs, ogs_array, t0s):
        self.ax = ax
        self.metric = metric
        self.metric_indx = metric_indxs[metric]
        self.metric_array = ogs_array[:, self.metric_indx]

        self.metric_min = np.nanmin(self.metric_array)
        self.metric_max = np.nanmax(self.metric_array)
        self.estimated_ymax = 7 * 1 / (self.metric_max - self.metric_min)
        self.artist = None

        self.t0s = np.array(t0s)  # the time associated with every row

    def initialize(self):
        """
        Initialize the plotting surface
        and extract the dynamic artist
        for your given axis
        """
        self.ax.set_ylim([0, 30])
        self.ax.set_xlim([self.metric_min, self.metric_max])
        axm = self.ax.twinx()
        axm.set_ylim([0, self.estimated_ymax])
        axm.set_xlim([self.metric_min, self.metric_max])
        axm.yaxis.set_visible(False)
        s, = axm.plot([], [], color="darkgrey", linewidth=2)
        self.artist = s

    def show_frame(self, j):
        """
        Show the animation for frame `j`

        """
        # Density
        m = self.metric_array[j][~np.isnan(self.metric_array[j])]
        try:
            kde = stats.kde.gaussian_kde(m)
            x = np.linspace(self.metric_min, self.metric_max, 100)
            y = kde(x)
            self.artist.set_data(x, y)  # here we update the artist object, I believe
        except np.linalg.linalg.LinAlgError:
            pass  # if the matrix is singular, can't make a kde

        # Hist
        self.ax.clear()
        self.ax.hist(m, bins=np.linspace(self.metric_min, self.metric_max, 25),
                     alpha=0.5, color="forestgreen")
        self.ax.set_xlim([self.metric_min, self.metric_max])
        self.ax.set_ylim([0, 30])
        self.ax.set_xlabel(self.metric)


class AnimateSFS(object):
    def __init__(self, ax, sfs_t0, sfs_bins, sfs_mean, sfs_se):
        self.ax = ax
        # Load Data
        self.sfs_t0 = sfs_t0
        self.sfs_bins = sfs_bins
        self.sfs_mean = sfs_mean
        self.sfs_se = sfs_se

        # Find Limits for Plot
        self.sfs_max = np.nanmax(self.sfs_mean)
        self.sfs_min = np.nanmin(self.sfs_mean)
        # Needed for animation
        self.lines = None

    def initialize(self, color):
        """
        Initialize the plotting surface
        and extract the dynamic artist
        for your given axis
        """
        self.ax.set_ylim([self.sfs_min, self.sfs_max])
        self.ax.set_xlim([0, 1])  # for the full SFS
        self.ax.set_xlabel("Site Frequency", fontsize=12)
        self.ax.set_ylabel("Occurances", fontsize=12)

        # Initialize Lines
        line, = self.ax.plot([], [],
                             linewidth=0.8,
                             marker='o',
                             color=color)
        self.lines = line

    def show_frame(self, j):
        """
        Show the animation for frame `j`

        """
        y = self.sfs_mean[j]
        self.lines.set_data(self.sfs_bins, y)


class AnimateLD(object):
    def __init__(self, ax, r2_t0, r2_bins, r2_mean, r2_se):
        self.ax = ax
        # Load Data
        self.r2_t0 = r2_t0
        self.r2_bins = r2_bins
        self.r2_mean = r2_mean
        self.r2_se = r2_se

        # Find Limits for Plot
        self.r2_max = np.nanmax(self.r2_mean)  # nan's exist early in simulation
        self.r2_min = np.nanmin(self.r2_mean)
        # Place for animation object(s)
        self.lines = None

    def initialize(self, color, nsnps):
        """
        Initialize the plotting surface
        and extract the dynamic artist
        for your given axis
        """
        self.ax.set_ylim([0, 1])
        self.ax.set_xlim([0, nsnps])
        self.ax.set_xlabel("Position (bp)", fontsize=12)
        self.ax.set_ylabel("Linkage Decay ($ r^2 $)", fontsize=12)

        # Initialize Lines
        line, = self.ax.plot([], [],
                             linewidth=0.8,
                             marker='o',
                             color=color)
        self.lines = line

    def show_frame(self, j):
        """
        Show the animation for frame `j`

        """
        y = self.r2_mean[j]
        self.lines.set_data(self.r2_bins, y)
