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


days_per_year = 365
years_per_major_tick = 10
def format_xaxis(value, tick_number,
                 days_per_year=days_per_year,
                 years_per_major_tick=years_per_major_tick):
    """
    Plot as Decade Number as opposed to Day Number
    """
    return int(round(value / (days_per_year * years_per_major_tick)))


def plot_individual_timecourse(op, og, epoch_df,
                               ax,
                               genetic_alpha=0.8,
                               genetic_names=None,
                               genetic_metrics=None,
                               focus_metric=None,
                               time_limits=None,
                               output_path=None):
    """
    Plot the prevalence and, optionally, genetic diversity
    statistics of a single forward-dream simulation through
    time
    """
    
    # Host
    ax.plot(op["t0"], op["HX"], color="blue", linewidth=0.8, label="Host All")
    ax.plot(op["t0"], op["HmX"], color="darkblue", linewidth=0.8, label="Host Mixed")
    
    # Vector
    ax.plot(op["t0"], op["VX"], color="red", linewidth=0.8, label="Vector All")
    ax.plot(op["t0"], op["VmX"], color="darkred", linewidth=0.8, label="Vector Mixed")
    
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
    ax.tick_params(axis='both', which='major', direction='in', length=10, labelsize=16)
    ax.tick_params(axis='both', which='minor', direction='in', length=8, labelsize=16)
    ax.xaxis.set_major_locator(plt.MultipleLocator(days_per_year * years_per_major_tick))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(days_per_year))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_xaxis))
    
    # Labels
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
                axm.set_ylabel(genetic_names[metric], fontsize=18)
                axm.tick_params(axis='y', which='major', direction='in', length=10, labelsize=16)
                axm.set_xlim(time_limits)
            else:
                axm = ax.twinx()
                axm.plot(og["t0"], og[metric], "forestgreen", alpha=genetic_alpha)
                axm.axes.get_yaxis().set_visible(False)
                axm.set_xlim(time_limits)
    
    return None

def plot_crash_response(metric, ot, epoch_df, d, e, ax):
    """
    Plot the detection and equilibrium times for a given metric
    for a given simulation
    """
    # Normalize time to start of crash
    start = epoch_df.loc["Crash", "t0"]
    t0 = ot["t0"] - start
    
    # Plot prevalence
    ax.fill_between(x=t0, y1=0, y2=ot["HX"], 
                    color="lightgrey", linewidth=0.8, 
                    label="Host All")
    ax.fill_between(x=t0, y1=0, y2=ot["VX"], 
                    color="darkgrey", linewidth=0.8, 
                    label="Vector All")
    
    # Delineate key epochs
    ax.axvline(epoch_df.loc["Crash", "t0"] - start, 
           color="grey", alpha=0.75,
           linewidth=3,
           zorder=2)
    ax.axvline(epoch_df.loc["CrashVar", "t0"] - start, 
               color="grey", alpha=0.75,
               linewidth=3,
               zorder=2)
    ax.axvline(epoch_df.loc["Recovery", "t0"] - start, 
               color="grey", alpha=0.75,
               linewidth=3,
               zorder=2)
    
    # Labels
    ax.set_ylabel("Prevalence")
    ax.set_xlabel("Time (years)")
    
    # Limits
    time_limits = (epoch_df.loc["InitVar", "t0"], epoch_df.loc["CrashVar", "t1"]) - start
    ax.set_xlim(time_limits)
    ax.set_ylim((0, 0.8))
    
    # Ticks, x-axis
    days_per_year = 365
    years_per_major_tick = 10
    ax.xaxis.set_major_locator(plt.MultipleLocator(days_per_year * years_per_major_tick))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(days_per_year))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, i : int(v / days_per_year)))
    
    # Twin axis
    axm = ax.twinx()

    # Plot Genetics
    axm.plot(t0, ot[metric], color="steelblue", linewidth=0.75, alpha=0.75)
    
    # Labels
    axm.set_ylabel(genetic_names[metric])
    
    # Limits
    axm.set_xlim(time_limits)

    # Demarcate detection
    axm.axvline(d[metric], 
                color="orange", alpha=0.75,
                linewidth=3,
                zorder=2)

    axm.axvline(e[metric], 
                color="red", alpha=0.75,
                linewidth=3,
                zorder=2)
    
    return None



def plot_recovery_response(metric, ot, epoch_df, d, e, ax):
    """
    Plot the detection and equilibrium times for a given metric
    for a given simulation
    """
    # Normalize time to start of crash
    start = epoch_df.loc["Recovery", "t0"]
    t0 = ot["t0"] - start
    
    # Plot prevalence
    ax.fill_between(x=t0, y1=0, y2=ot["HX"], 
                    color="lightgrey", linewidth=0.8, 
                    label="Host All")
    ax.fill_between(x=t0, y1=0, y2=ot["VX"], 
                    color="darkgrey", linewidth=0.8, 
                    label="Vector All")
    
    # Delineate key epochs
    ax.axvline(epoch_df.loc["CrashVar", "t0"] - start, 
           color="grey", alpha=0.75,
           linewidth=3,
           zorder=2)
    ax.axvline(epoch_df.loc["Recovery", "t0"] - start, 
               color="grey", alpha=0.75,
               linewidth=3,
               zorder=2)
    
    # Labels
    ax.set_ylabel("Prevalence")
    ax.set_xlabel("Time (years)")
    
    # Limits
    time_limits = (epoch_df.loc["CrashVar", "t0"], epoch_df.loc["Recovery", "t1"]) - start
    ax.set_xlim(time_limits)
    ax.set_ylim((0, 0.8))
    
    # Ticks, x-axis
    days_per_year = 365
    years_per_major_tick = 10
    ax.xaxis.set_major_locator(plt.MultipleLocator(days_per_year * years_per_major_tick))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(days_per_year))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, i : int(v / days_per_year)))
    
    # Twin axis
    axm = ax.twinx()

    # Plot Genetics
    axm.plot(t0, ot[metric], color="steelblue", linewidth=0.75, alpha=0.75)
    
    # Labels
    axm.set_ylabel(genetic_names[metric])
    
    # Limits
    axm.set_xlim(time_limits)

    # Demarcate detection
    axm.axvline(d[metric], 
                color="orange", alpha=0.75,
                linewidth=3,
                zorder=2)

    axm.axvline(e[metric], 
                color="red", alpha=0.75,
                linewidth=3,
                zorder=2)
    
    return None

