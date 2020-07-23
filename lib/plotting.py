import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy
import statsmodels.formula.api as smf
from lib.preferences import *


def get_boxplot_bands(ax, ols, x, alpha=0.05):
    """
    Produce a linear regression
    that will map to a boxplot
    """
    b = ols.params[0]
    m = ols.params[1]
    ols_line = lambda x: b + m * x
    
    npts = 100
    x_ = np.linspace(0.0, 1.0, npts)  # full prevalence range
    
    # convert to plotting x-axis values
    x_min, x_max = x.min(), x.max()
    xp_min, xp_max = ax.get_xticks().min(), ax.get_xticks().max()
    coeff = (xp_max - xp_min)/(x_max - x_min)
    x_convert = lambda x: (x - x_min)*coeff
    xp_ = x_convert(x_)
    
    y_ = ols_line(x_)
    
    # Confidence / Prediction Intervals
    SSx = np.sum((x - np.mean(x)) ** 2)
    n = ols.nobs
    rse = np.sqrt(ols.mse_resid)

    conf_se = rse * np.sqrt(1. / n + (x_ - np.mean(x)) ** 2 / SSx)
    pred_se = rse * np.sqrt(1 + 1. / n + (x_ - np.mean(x)) ** 2 / SSx)

    tval = scipy.stats.t.isf(alpha / 2., ols.df_resid)

    conf_u = y_ + tval * conf_se
    conf_l = y_ - tval * conf_se
    pred_u = y_ + tval * pred_se
    pred_l = y_ - tval * pred_se
    
    return xp_, y_, conf_l, conf_u, pred_l, pred_u


def regress_boxplot(metric, x_h, df, ax, palette):
    """
    Create a boxplot with a regression line
    """
    
    # Boxplot
    sns.boxplot(x=x_h, y=metric,
                data=df,
                palette=palette,
                ax=ax)
    ylims = ax.get_ylim() # store for later
    xlims = ax.get_xlim()
    
    # Linear Model
    ols = smf.ols(metric + "~" + x_h, df).fit()
    x_, y_, conf_l, conf_u, pred_l, pred_u = get_boxplot_bands(ax, ols, df[x_h])
    ax.plot(x_, y_, color="darkgrey", zorder=-2)
    ax.fill_between(x=x_, y1=conf_l, y2=conf_u, color='grey', alpha=0.5, zorder=-2)
    ax.fill_between(x=x_, y1=pred_l, y2=pred_u, color='grey', alpha=0.25, zorder=-2)
    ax.set_ylim(ylims)
    ax.set_xlim(xlims)
    # Legend
    ax.annotate(xy=(0.05, 0.85), xycoords='axes fraction',
                s="$ r^2 = $ %.02f" % ols.rsquared, fontsize=10,
                bbox=dict(facecolor='white', edgecolor='grey'), zorder=5)

    return None


def regress_violinplot(metric, x_h, df, ax, palette):
    """
    Create a boxplot with a regression line
    """
    
    # Boxplot
    sns.violinplot(x=x_h, y=metric,
                   data=df,
                   palette=palette,
                   ax=ax)
    ylims = ax.get_ylim() # store for later
    xlims = ax.get_xlim()
    
    # Linear Model
    ols = smf.ols(metric + "~" + x_h, df).fit()
    x_, y_, conf_l, conf_u, pred_l, pred_u = get_boxplot_bands(ax, ols, df[x_h])
    ax.plot(x_, y_, color="darkgrey", zorder=-2)
    ax.fill_between(x=x_, y1=conf_l, y2=conf_u, color='grey', alpha=0.5, zorder=-2)
    ax.fill_between(x=x_, y1=pred_l, y2=pred_u, color='grey', alpha=0.25, zorder=-2)
    ax.set_ylim(ylims)
    ax.set_xlim(xlims)
    # Legend
    ax.annotate(xy=(0.05, 0.85), xycoords='axes fraction',
                s="$ r^2 = $ %.02f" % ols.rsquared, fontsize=10,
                bbox=dict(facecolor='white', edgecolor='grey'), zorder=5)

    return None


def regress_scatterplot(metric, x_h, ot, ax, color, **kwargs):
    """
    Create a scatterplot with a regression line
    
    """
    # Scatterplot
    ax.scatter(x=x_h, y=metric,
               c=[color]*len(ot),
               data=ot, **kwargs)
    ylims = ax.get_ylim() # store for later
    xlims = ax.get_xlim()

    # Linear Model
    ols = smf.ols(metric + "~" + x_h, ot).fit()
    x_, y_, conf_l, conf_u, pred_l, pred_u = get_boxplot_bands(ax, ols, ot[x_h])
    ax.plot(x_, y_, color="darkgrey", zorder=-2)
    ax.fill_between(x=x_, y1=conf_l, y2=conf_u, color='grey', alpha=0.5, zorder=-2)
    ax.fill_between(x=x_, y1=pred_l, y2=pred_u, color='grey', alpha=0.25, zorder=-2)
    ax.set_ylim(ylims)
    ax.set_xlim(xlims)
    # Legend
    ax.annotate(xy=(0.05, 0.875), xycoords='axes fraction',
                s="$ r^2 = $ %.02f" % ols.rsquared, fontsize=10,
                bbox=dict(facecolor='white', edgecolor='grey'), zorder=5)


# ================================================================================ #
# Trajectory plots; longitudinal analysis of prevalence and genetic diversity
# statistics
# ================================================================================ #


def prevalence_trajectory_plot(ot, epoch_df, ax,
                               col_dt,
                               norm_t0=None,
                               indicate_epochs=None,
                               indicate_equilibriums=None,
                               time_limits=None,
                               years_per_major_tick=5):
    """
    Plot the trajectory of a genetic diversity
    statistic through time
    
    
    """
    
    # Normalize time to desired epoch
    if norm_t0 is not None:
        start = epoch_df.loc[norm_t0]
    else:
        start = epoch_df.loc["init", "t0"]
    t0 = ot["t0"] - start
    
    
    # Host Prevalence
    ax.plot(t0, ot["HX"], color=col_dt["HX"], linewidth=0.8, label="Host All")
    ax.plot(t0, ot["HmX"], color=col_dt["HmX"], linewidth=0.8, label="Host Mixed")

    # Vector Prevalence
    ax.plot(t0, ot["VX"], color=col_dt["VX"], linewidth=0.8, label="Vector All")
    ax.plot(t0, ot["VmX"], color=col_dt["VmX"], linewidth=0.8, label="Vector Mixed")
    
        
    # Set limits
    ax.set_ylim((0, 1.0))
    ax.set_yticks(np.arange(0, 1.2, 0.2))
    ax.set_xlim(time_limits - start)
    
    # Set ticks, x-axis
    days_per_year = 365
    ax.xaxis.set_major_locator(plt.MultipleLocator(days_per_year * years_per_major_tick))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(days_per_year))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, i : int(v / days_per_year)))
    
    # Delineate epoch boundaries
    if indicate_epochs is not None:
        for epoch in indicate_epochs:
            ax.axvline(epoch_df.loc[epoch] - start, 
                       color="grey", alpha=0.75,
                       linewidth=3, zorder=0)
            
    # Indicate equilibriums
    if indicate_equilibriums is not None:
        for epoch in indicate_equilibriums:
            x = [epoch_df.loc[(epoch, "t0")] - start, epoch_df.loc[(epoch, "t1")] - start]
            h = [epoch_df.loc[(epoch, "x_h")], epoch_df.loc[(epoch, "x_h")]]
            v = [epoch_df.loc[(epoch, "x_v")], epoch_df.loc[(epoch, "x_v")]]
            ax.plot(x, h, linestyle="dashed", color=col_dt["HX"])
            ax.plot(x, v, linestyle="dashed", color=col_dt["VX"])
            
    return None




def prevalence_trajectory_average_plot(ot_mu, ot_se, epoch_df, ax,
                                       col_dt,
                                       norm_t0=None,
                                       indicate_epochs=None,
                                       indicate_equilibriums=None,
                                       time_limits=None,
                                       years_per_major_tick=5):
    """
    Plot the trajectory of a genetic diversity
    statistic through time
    
    
    """
    
    # Normalize time to desired epoch
    if norm_t0 is not None:
        start = epoch_df.loc[norm_t0]
    else:
        start = epoch_df.loc["init", "t0"]
    t0 = ot_mu["t0"] - start
    
    
    # Plot prevalence
    n_se = 1.96
    se_alpha = 0.25
    metrics = ["HX", "HmX", "VX", "VmX"]
    dt = {"HX": "Host All", "VX": "Vector All", "HmX": "Host Mixed", "VmX": "Vector Mixed"}
    for metric in metrics:
        ax.plot(t0, ot_mu[metric], color=col_dt[metric], linewidth=0.8, label=dt[metric])
        ax.fill_between(x=t0,
                        y1=ot_mu[metric] - ot_se[metric] * n_se,
                        y2=ot_mu[metric] + ot_se[metric] * n_se,
                        color=col_dt[metric], alpha=se_alpha)
    
    # Set limits
    ax.set_ylim((0, 1.0))
    ax.set_yticks(np.arange(0, 1.2, 0.2))
    ax.set_xlim(time_limits - start)
    
    # Set ticks, x-axis
    days_per_year = 365
    ax.xaxis.set_major_locator(plt.MultipleLocator(days_per_year * years_per_major_tick))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(days_per_year))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, i : int(v / days_per_year)))
    
    # Delineate epoch boundaries
    if indicate_epochs is not None:
        for epoch in indicate_epochs:
            ax.axvline(epoch_df.loc[epoch] - start, 
                       color="grey", alpha=0.75,
                       linewidth=3, zorder=0)
            
    # Indicate equilibriums
    if indicate_equilibriums is not None:
        for epoch in indicate_equilibriums:
            x = [epoch_df.loc[(epoch, "t0")] - start, epoch_df.loc[(epoch, "t1")] - start]
            h = [epoch_df.loc[(epoch, "x_h")], epoch_df.loc[(epoch, "x_h")]]
            v = [epoch_df.loc[(epoch, "x_v")], epoch_df.loc[(epoch, "x_v")]]
            print(x, h)
            ax.plot(x, h, linestyle="dashed", color=col_dt["HX"])
            ax.plot(x, v, linestyle="dashed", color=col_dt["VX"])
            
    return None




def genetic_trajectory_plot(metric, ot, epoch_df,
                            color, ax,
                            norm_t0=None,
                            indicate_epochs=None,
                            time_limits=None,
                            t_detection=None,
                            t_equilibrium=None,
                            alpha=1.0,
                            years_per_major_tick=5):
    """
    Plot the trajectory of a genetic diversity
    statistic through time
    
    
    """
    
    # Trim to reduce computational expense
    ot = ot.query("@time_limits[0] <= t0 <= @time_limits[1]")
    
    # Normalize time to desired epoch
    if norm_t0 is not None:
        start = epoch_df.loc[norm_t0]
    else:
        start = epoch_df.loc["init", "t0"]
    t0 = ot["t0"] - start
    
    
    # Plot genetic metric
    ax.plot(t0, ot[metric], 
            color=color, linewidth=0.75, 
            alpha=alpha, zorder=1)
    
    
    # Demarcate detection & equilibrium
    if t_detection is not None:
        ax.axvline(t_detection[metric], 
                   color=color, alpha=0.75,
                   linewidth=3, linestyle="dashed",
                   zorder=2)
        
    if t_equilibrium is not None:
        ax.axvline(t_equilibrium[metric], 
                   color=color, alpha=0.75,
                   linewidth=3, linestyle="solid",
                   zorder=2)
        
    # Set limits, x-axis
    ax.set_xlim(time_limits - start)
    
    # Set ticks, x-axis
    days_per_year = 365
    ax.xaxis.set_major_locator(plt.MultipleLocator(days_per_year * years_per_major_tick))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(days_per_year))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, i : int(v / days_per_year)))
    
    # Twin axis
    ax.patch.set_visible(False)
    axm = ax.twinx()
    axm.set_zorder(ax.get_zorder() - 1)
    
    # Plot prevalence
    axm.fill_between(x=t0, y1=0, y2=ot["HX"], 
                    color="lightgrey", linewidth=0.8, 
                    label="Host All")
    axm.fill_between(x=t0, y1=0, y2=ot["VX"], 
                    color="darkgrey", linewidth=0.8, 
                    label="Vector All")
    
    # Delineate epoch boundaries
    if indicate_epochs is not None:
        for epoch in indicate_epochs:
            axm.axvline(epoch_df.loc[epoch] - start, 
                        color="grey", alpha=0.75,
                        linewidth=3,
                        zorder=0)
    
    
    # Hide y-labels
    axm.set_yticklabels("")
    
    # Limits
    axm.set_xlim(time_limits - start)
    axm.set_ylim((0, 1.0))
    axm.set_yticks(np.arange(0, 1.2, 0.2))
    
    return None




def genetic_trajectory_average_plot(metric, ot_mu, ot_se, epoch_df,
                                    color, ax,
                                    norm_t0=None,
                                    indicate_epochs=None,
                                    time_limits=None,
                                    t_detection=None,
                                    t_equilibrium=None,
                                    years_per_major_tick=5):
    """
    Plot the trajectory of a genetic diversity
    statistic through time
    
    
    """
    
    # Normalize time to desired epoch
    if norm_t0 is not None:
        start = epoch_df.loc[norm_t0]
    else:
        start = epoch_df.loc["init", "t0"]
    t0 = ot_mu["t0"] - start
    
    
    # Plot genetic metric
    n_se = 1.96
    se_alpha = 0.25
    ax.plot(t0, ot_mu[metric], color=color, linewidth=1.0)
    ax.fill_between(x=t0,
                    y1=ot_mu[metric] - ot_se[metric] * n_se,
                    y2=ot_mu[metric] + ot_se[metric] * n_se,
                    color=color, alpha=se_alpha)
    
    
    # Demarcate detection & equilibrium
    if t_detection is not None:
        ax.axvline(t_detection[metric], 
                   color=color, alpha=0.75,
                   linewidth=3, linestyle="dashed",
                   zorder=2)
        
    if t_equilibrium is not None:
        ax.axvline(t_equilibrium[metric], 
                   color=color, alpha=0.75,
                   linewidth=3, linestyle="solid",
                   zorder=2)
        
    # Set limits, x-axis
    ax.set_xlim(time_limits - start)
    
    # Set ticks, x-axis
    days_per_year = 365
    ax.xaxis.set_major_locator(plt.MultipleLocator(days_per_year * years_per_major_tick))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(days_per_year))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, i : int(v / days_per_year)))
    
    # Twin axis
    ax.patch.set_visible(False)
    axm = ax.twinx()
    axm.set_zorder(ax.get_zorder() - 1)
    
    # Plot prevalence
    axm.fill_between(x=t0, y1=0, y2=ot_mu["HX"], 
                    color="lightgrey", linewidth=0.8, 
                    label="Host All")
    axm.fill_between(x=t0, y1=0, y2=ot_mu["VX"], 
                    color="darkgrey", linewidth=0.8, 
                    label="Vector All")
    
    # Delineate epoch boundaries
    if indicate_epochs is not None:
        for epoch in indicate_epochs:
            axm.axvline(epoch_df.loc[epoch] - start, 
                        color="grey", alpha=0.75,
                        linewidth=3,
                        zorder=0)
    
    
    # Hide y-labels
    axm.set_yticklabels("")
    
    # Limits
    axm.set_xlim(time_limits - start)
    axm.set_ylim((0, 1.0))
    axm.set_yticks(np.arange(0, 1.2, 0.2))
    
    return None

