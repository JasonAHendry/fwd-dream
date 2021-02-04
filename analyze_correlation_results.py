import sys
import getopt
from lib.correlation import *
from lib.preferences import *


# PARSE COMMAND-LINE
try:
    opts, args = getopt.getopt(sys.argv[1:], ":e:")
except getopt.GetoptError:
    print("Option Error. Please conform to:")
    print("-e <str> -p <str>")

for opt, value in opts:
    if opt == "-e":
        expt_name = value
        expt_path = os.path.join("results", expt_name)
        output_path = os.path.join("analysis", expt_name)
        fig_path = os.path.join(output_path, "figs")
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        if not os.path.isdir(fig_path):
            os.mkdir(fig_path)
        print("Preparing to analyze", expt_name)
        print(" Experiment Path:", expt_path)
        print(" Output Path:", output_path)
        print(" Figure Path:", fig_path)
    else:
        print("Parameter %s not recognized." % opt)
        sys.exit(2)


# LOAD SIMULATIONS
print("Loading Simulations...")
track_ibd = True
if track_ibd:
    genetic_metrics += ibd_metrics
s_paths = [os.path.join(expt_path, s) for s in os.listdir(expt_path)]
s_paths = [s for s in s_paths if os.path.isdir(s)]
s_paths_complete = [s for s in s_paths if "Endpoint" in os.listdir(s)]
n_simulations = len(s_paths_complete)
print("Number of Simulations:", len(s_paths))
print("... without extinction:", n_simulations)
simulations = [CorrelationResult(s) for s in s_paths_complete]

# TRAJECTORY PLOTS
print("Plotting prevalence trajectories...")
trajectory_dirs = os.path.join(fig_path, "trajectories")
if not os.path.isdir(trajectory_dirs):
    os.mkdir(trajectory_dirs)
for s in simulations:
    s.plot_prevalence_trajectory(make_pdf=True, output_path=trajectory_dirs)
    s.plot_prevalence_trajectory(make_pdf=True, time_limits=[0, 3650],
                                 output_path=trajectory_dirs)

# CORRELATION SCATTER PLOTS
print("Creating genetic diversity DataFrame...")
genetic_metrics = remove_metrics(rmv=['n_fixed_ref', 'n_fixed_alt', 'frac_uniq_barcodes'],
                                 metrics=genetic_metrics)
metric_ll = [[s.name, s.nh, s.nv, s.ne_theory, s.x_h_theory, s.x_h_obs,
              s.n_re_genomes, s.n_mixed_genomes, s.frac_mixed_genomes,
              s.n_re_samples, s.n_mixed_samples, s.frac_mixed_samples,
              s.mean_k,
              s.n_barcodes, s.single_barcodes,
              s.n_variants, s.n_segregating, s.n_singletons,
              s.pi, s.theta, s.tajd,
              s.avg_frac_ibd, s.avg_l_ibd, s.avg_n_ibd] for s in simulations]
metric_df = pd.DataFrame(metric_ll, columns=["name", "nh", "nv", "ne_theory", "x_h_theory", "x_h_obs"] + genetic_metrics)
metric_df.sort_values("x_h_theory", inplace=True)
# Filtering
metric_df = metric_df.query("x_h_theory < 0.89")
print(" No. rows (expts):", metric_df.shape[0])
print(" No. cols (metrics):", metric_df.shape[1])
metric_df.to_csv(os.path.join(output_path, "metric_df.csv"), index=False)
# Colours
metric_cols = create_metric_colours(pal="Paired",
                                    genetic_metrics=genetic_metrics,
                                    genetic_grps=genetic_grps)
# Individual Plots
print("Plotting individual scatterplots...")
scatter_dir = "scatterplots"
scatter_path = os.path.join(fig_path, scatter_dir)
if not os.path.isdir(scatter_path):
    os.mkdir(scatter_path)
if not os.path.isdir(os.path.join(scatter_path, "x_h_theory")):
    os.mkdir(os.path.join(scatter_path, "x_h_theory"))
if not os.path.isdir(os.path.join(scatter_path, "x_h_obs")):
    os.mkdir(os.path.join(scatter_path, "x_h_obs"))
for i, metric in enumerate(genetic_metrics):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    metric_scatterplot(metric_df, x_h="x_h_theory", metric=metric,
                       genetic_names=genetic_names, col_dt=metric_cols,
                       ax=ax)
    fig.savefig(os.path.join(scatter_path, "x_h_theory", metric + ".png"),
                bbox_inches="tight", pad_inches=0.5, dpi=1000)
    plt.close(fig)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    metric_scatterplot(metric_df, x_h="x_h_obs", metric=metric,
                       genetic_names=genetic_names, col_dt=metric_cols,
                       ax=ax)
    fig.savefig(os.path.join(scatter_path, "x_h_obs", metric + ".png"),
                bbox_inches="tight", pad_inches=0.5, dpi=1000)
    plt.close(fig)
print("Done.")

# metrics vs. `x_h_theory`
print("Plotting Full Correlation Array...")
n_metrics = len(genetic_metrics)
n_cols = 3
n_rows = -(-n_metrics // n_cols)
fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
fig.subplots_adjust(hspace=0.15)
for i, metric in enumerate(genetic_metrics):
    print(" ", metric)
    r = i / n_cols
    c = i - r * n_cols
    metric_scatterplot_array(metric_df, x_h="x_h_theory",
                             col_dt=metric_cols, xlabel=r == n_rows - 1,
                             metric=metric, genetic_names=genetic_names,
                             ax=ax[r, c])

fig.savefig(os.path.join(fig_path, "x_h_theory.png"), dpi=1000)
plt.close(fig)
# metrics vs. `x_h_obs`
fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
fig.subplots_adjust(hspace=0.15)
for i, metric in enumerate(genetic_metrics):
    r = i / n_cols
    c = i - r * n_cols
    metric_scatterplot_array(metric_df, x_h="x_h_obs",
                             col_dt=metric_cols, xlabel=r == n_rows - 1,
                             metric=metric, genetic_names=genetic_names,
                             ax=ax[r, c])
fig.savefig(os.path.join(fig_path, "x_h_obs.png"), dpi=1000)
plt.close(fig)
print("Done.")
# WITH TIGHT METRICS
tight_metrics = ['n_re_samples',
                 'frac_mixed_samples', 'mean_k',
                 'n_segregating', 'n_singletons',
                 'pi', 'theta', 'tajd']
# metrics vs. `x_h_theory`
print("Plotting Tight Correlation Array...")
n_metrics = len(tight_metrics)
n_cols = 4
n_rows = -(-n_metrics // n_cols)
fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
fig.subplots_adjust(hspace=0.15)
axes = ax.flatten()
for i, metric in enumerate(tight_metrics):
    print(" ", metric)
    r = i / n_cols
    c = i - r * n_cols
    metric_scatterplot_array(metric_df, x_h="x_h_theory",
                             col_dt=metric_cols, xlabel=r == n_rows - 1,
                             metric=metric, genetic_names=genetic_names,
                             ax=ax[r, c])
fig.savefig(os.path.join(fig_path, "tight_x_h_theory.png"))
plt.close(fig)
# metrics vs. `x_h_obs`
fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
fig.subplots_adjust(hspace=0.15)
axes = ax.flatten()
for i, metric in enumerate(tight_metrics):
    print(" ", metric)
    r = i / n_cols
    c = i - r * n_cols
    metric_scatterplot_array(metric_df, x_h="x_h_obs",
                             col_dt=metric_cols, xlabel=r == n_rows - 1,
                             metric=metric, genetic_names=genetic_names,
                             ax=ax[r, c])
fig.savefig(os.path.join(fig_path, "tight_x_h_obs.png"))
plt.close(fig)


# CORRELATION MATRIX
print("Plotting Correlation Matrix...")
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
cmap = sns.diverging_palette(220,10, as_cmap=True)
sns.heatmap(metric_df.corr(),
            cmap=cmap,
            square=True,
            cbar_kws={"shrink": .8},
            annot=True,
            linewidths=0.5,  # lines to separate cells
            linecolor='black',
            ax=ax)
ax.set_yticklabels(ax.yaxis.get_majorticklabels(), rotation=45, ha="right")
ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
# Color Bar Details
cbar = ax.collections[0].colorbar  # convoluted way to get to the colorbar object
cbar.ax.set_ylabel("Pearson's $ R $",
                   rotation=-90, fontsize=14,
                   labelpad=25,  # move it a bit away from the axis
                  )
cbar.outline.set_linewidth(1.5)  # modify some features of the outline
cbar.outline.set_edgecolor("black")
cbar.set_ticks(np.arange(-1, 1.5, 0.5))
cbar.ax.hlines(y=np.arange(-1, 1.5, 0.5), xmin=0, xmax=1, linewidth=1)
cbar.ax.minorticks_on()
cbar.ax.tick_params(axis='y', direction='out', length=8, width=1)
fig.savefig(os.path.join(fig_path, "matrix.png"))
print("Done.")


# r2 DECAY (could be put in function in future)
# binning settings
print("Plotting r2 decay...")
n_bins_r2 = 20
bins_r2 = np.linspace(0, 1000, n_bins_r2 + 1)
bin_midpoints_r2 = (bins_r2[1:] + bins_r2[:-1])/2.0
# create r2 array
r2_array = np.zeros((n_simulations, n_bins_r2))
x_hs = np.zeros(n_simulations)
for i, simulation in enumerate(simulations):
    r2, d = simulation.get_r2()
    binned_r2, _, _ = scipy.stats.binned_statistic(x=d,
                                                   values=r2,
                                                   statistic='mean',
                                                   bins=bins_r2)
    r2_array[i] = binned_r2
    x_hs[i] = simulation.x_h_theory
x_hs = np.unique(x_hs)
# compute mean across bins
r2_df = pd.DataFrame(r2_array, columns=[str(r) for r in bin_midpoints_r2])
r2_df["x_h_theory"] = [s.x_h_theory for s in simulations]
r2_means = np.array(r2_df.groupby("x_h_theory").mean())
# plot
fig, ax = plt.subplots(1, 1)
r2_cols = sns.color_palette('plasma', len(x_hs))
r2_col_dt = {x_h: col for x_h, col in zip(x_hs, r2_cols)}
for i, x_h in enumerate(x_hs):
    ax.plot(bin_midpoints_r2, r2_means[i],
            marker='o', ms=5,
            color=r2_col_dt[x_h],
            label="%.02f" % x_h)
    # Ticks
    ax.set_xticks(bins_r2 + 2.5)
    r2_bin_size = np.diff(bin_midpoints_r2).max()
    xtick_labs = ["%.0f - %.0f" % (r - r2_bin_size/2.0, r + r2_bin_size/2.0) for r in bin_midpoints_r2]
    ax.set_xticklabels(xtick_labs, rotation=-45, ha="center", va="baseline")
    ax.grid(linestyle='--')
    # Text
    ax.set_ylabel("$ r^2 $", fontsize=14)
    ax.set_xlabel("Distance Between SNPs (bp)", fontsize=14)
    ax.set_title("Linkage Disequilibrium Decay \n Across $ X_H $", fontsize=14)
l = ax.legend(fontsize=12, ncol=2,
              bbox_to_anchor=(1, 1),
              frameon=True, title="Host Prevalence, $X_H$")
l = plt.setp(l.get_title(), fontsize=12)
fig.savefig(os.path.join(fig_path, "r2.png"), bbox_inches="tight", pad_inches=1.0)
print("Done.")


# SITE FREQUENCY SPECTRUM
# binning settings
n_bins_sfs = 10
bins_sfs = np.linspace(0, 1, n_bins_sfs + 1)
bin_midpoints_sfs = (bins_sfs[1:] + bins_sfs[:-1])/2.0
# create sfs array
sfs_array = np.zeros((n_simulations, n_bins_sfs))
x_hs = np.zeros(n_simulations)
for i, simulation in enumerate(simulations):
    n = simulation.n_re_genomes
    unfolded_sfs = simulation.get_sfs()
    binned_sfs, _, _ = scipy.stats.binned_statistic(x=np.arange(1, n) / float(n),
                                                    values=unfolded_sfs,
                                                    statistic='sum',
                                                    bins=bins_sfs)
    sfs_array[i] = binned_sfs
    x_hs[i] = simulation.x_h_theory
x_hs = np.unique(x_hs)
# compute mean across bins
sfs_df = pd.DataFrame(sfs_array, columns = [str(r) for r in bin_midpoints_sfs])
sfs_df["x_h_theory"] = [s.x_h_theory for s in simulations]
sfs_means = np.array(sfs_df.groupby("x_h_theory").mean())
# plot
fig, ax = plt.subplots(1, 1)
sfs_cols = sns.color_palette('viridis', len(x_hs))
sfs_col_dt = {x_h: col for x_h, col in zip(x_hs, sfs_cols)}
for i, x_h in enumerate(x_hs):
    ax.plot(bin_midpoints_sfs, sfs_means[i],
            marker='o', ms=5,
            color=sfs_col_dt[x_h],
            label="%.02f" % x_h)
    # Ticks
    ax.set_xticks(bin_midpoints_sfs)
    sfs_bin_size = np.diff(bin_midpoints_sfs).max()
    xtick_labs = ["%.02f - %.02f" % (s - sfs_bin_size/2.0, s + sfs_bin_size/2.0) for s in bin_midpoints_sfs]
    ax.set_xticklabels(xtick_labs, rotation=-45, ha="left", va="baseline")
    ax.grid(linestyle='--')
    # Text
    ax.set_ylabel("$ \zeta_i $", fontsize=14)
    ax.set_xlabel("Frequency, $i$", fontsize=14)
    ax.set_title("Site-frequency Spectrum \n Across $ X_H $", fontsize=14)
l = ax.legend(fontsize=12, ncol=2,
              bbox_to_anchor=(1, 1),
              frameon=True, title="Host Prevalence, $X_H$")
l = plt.setp(l.get_title(), fontsize=12)
fig.savefig(os.path.join(fig_path, "sfs.png"), bbox_inches="tight", pad_inches=1.0)
