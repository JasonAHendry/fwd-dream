import os
import numpy as np
import pandas as pd
from itertools import islice
from scipy import stats


def window(seq, n=2):
    """
    Returns a sliding window (of width n) over data from the iterable
       s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   
    
    This function is from Daniel DePaolo, here:
    https://stackoverflow.com/questions/6822725/rolling-or-sliding-window-iterator
    
    """
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


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


def calc_detection_time(ot, epoch_df,
                        initial_epoch, respond_epoch,
                        analysis_metrics, 
                        alpha=0.05, robustness=None,
                        n_roll=None,
                        verbose=False):
    """
    Calculate the time until detection for a single
    simulation, across `analysis` metrics
    
    """
    # Subset to Relevant Epochs
    initial = subset_to_epoch(ot, epoch_df, initial_epoch)
    respond = subset_to_epoch(ot, epoch_df, respond_epoch)
    
    # Compute `bounds` for Detection
    quantiles = [alpha/2, 1 - alpha/2]
    bounds = initial[analysis_metrics].quantile(quantiles)
    
    # Smooth with rolling window, if `rolling`
    if type(n_roll) is int:
        respond = respond.rolling(n_roll).mean().dropna()
        
    # Extract Time Column, Normalize to Start of Respond
    t0 = respond["t0"].values
    t0 -= t0[0]
    
    # Is metric `outside` of `bounds`?
    respond = respond[analysis_metrics]
    below = respond.le(bounds.loc[quantiles[0]])
    above = respond.gt(bounds.loc[quantiles[1]])
    outside = below | above
    
    # Convert to detection time
    if type(robustness) is int:
        k = (True,) * robustness
        try:
            detect_t0 = outside.apply(lambda x: t0[np.where([w == k for w in window(x, n=robustness)])[0].min()], 0)
        except ValueError:
            print("Failed to determine detection time for all metrics.")
            detect_t0 = np.nan
    else:
        try:
            detect_t0 = outside.apply(lambda x: t0[np.where(x)[0]].min(), 0)
        except ValueError:
            print("Failed to determine detection time for all metrics.")
            detect_t0 = np.nan
    
    return detect_t0



def calc_equilibrium_time(ot, epoch_df,
                          initial_epoch, respond_epoch, equilibrium_epoch,
                          analysis_metrics, 
                          alpha=0.5, robustness=None,
                          n_roll=None,
                          verbose=False):
    """
    Calculate the time until detection for a single
    simulation, across `analysis` metrics
    
    """
    # Subset to Relevant Epochs
    initial = subset_to_epoch(ot, epoch_df, initial_epoch)
    respond = subset_to_epoch(ot, epoch_df, respond_epoch)
    equil = subset_to_epoch(ot, epoch_df, equilibrium_epoch)
    
    # Compute `bounds` for Detection
    quantiles = [alpha/2, 1 - alpha/2]
    bounds = equil[analysis_metrics].quantile(quantiles)
    
    
    # Smooth with rolling window, if `rolling`
    if type(n_roll) is int:
        respond = respond.rolling(n_roll).mean().dropna()
        
    # Extract Time Column, Normalize to Start of Respond
    t0 = respond["t0"].values
    t0 -= t0[0]
    
    # Is metric `within` of `bounds`?
    respond = respond[analysis_metrics]
    below = respond.le(bounds.loc[quantiles[1]])
    above = respond.gt(bounds.loc[quantiles[0]])
    inside = below & above
    
    # Convert to detection time
    if type(robustness) is int:
        k = (True,) * robustness
        try:
            detect_t0 = inside.apply(lambda x: t0[np.where([w == k for w in window(x, n=robustness)])[0].min()], 0)
        except ValueError:
            print("Failed to determine equilibrium time for all metrics.")
            detect_t0 = np.nan
    else:
        try:
            detect_t0 = inside.apply(lambda x: t0[np.where(x)[0]].min(), 0)
        except ValueError:
            print("Failed to determine equilibrium time for all metrics.")
            detect_t0 = np.nan
    
    return detect_t0



