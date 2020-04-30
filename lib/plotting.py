import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy
import statsmodels.formula.api as smf


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