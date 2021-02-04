import os
import numpy as np
import pandas as pd


def create_edges_set(config, epoch_df, data_type, verbose=False):
    """
    For a given experiment, create a set of
    edges of each Epoch that will define boundaries
    used for the binning of longitudinal data
    
    Parameters
        config : configparser.ConfigParser
            ConfigParser object pointing to the
            parameter file containing the Epochs
            of interest.
        epoch_df : DataFrame, shape (n_epochs, 7)
            Data frame specifying the realised time
            limits for each epoch.
        data_type : str
            Either "div" or "prev"; specifies
            whether you want to create edges for
            genetic diversity ("div") or prevalence
            ("prev") data.
        obs_per_bin : int
            The number of observations, on average,
            to include in a bin.
            
    """
    
    # Define type of data
    data = "%s_samp_freq" % data_type
    
    # Initialisation
    edges = []
    freq = config.getint("Sampling", data)
    start = epoch_df.loc[("init", "t0")]
    end = epoch_df.loc[("init", "t1")]
    edges.append((start, end, freq))
    
    # Get epochs
    epoch_sections = [s for s in config.sections() if s.startswith("Epoch_")]
    
    # Iterate over epochs
    for epoch in epoch_sections:
        # Parse
        epoch_name = epoch.split("_")[1]
        start = epoch_df.loc[(epoch_name, "t0")]
        end = epoch_df.loc[(epoch_name, "t1")]

        # Adjust 
        if config.has_option(epoch, data):
            efreq = config.getint(epoch, data)

            if config.has_option(epoch, "%s_samp_t" % data_type):
                tend = config.getfloat(epoch, "%s_samp_t" % data_type)
                edges.append((start, tend, efreq))
                edges.append((tend, end, freq))
            else:
                edges.append((start, end, efreq))
                freq = efreq
        else:
            edges.append((start, end, freq))
        
    return edges


def create_bin_boundaries(config, epoch_df, data_type, obs_per_bin, verbose=False):
    """
    For a given experiment, create time boundaries
    used for the binning of longitudinal data
    
    Parameters
        config : configparser.ConfigParser
            ConfigParser object pointing to the
            parameter file containing the Epochs
            of interest.
        epoch_df : DataFrame, shape (n_epochs, 7)
            Data frame specifying the realised time
            limits for each epoch.
        data_type : str
            Either "div" or "prev"; specifies
            whether you want to create edges for
            genetic diversity ("div") or prevalence
            ("prev") data.
        obs_per_bin : int
            The number of observations, on average,
            to include in a bin.
            
    """
    
    edges = create_edges_set(config, epoch_df, data_type)
    
    boundaries = []
    for edge in edges:
        start, end, freq = edge
        bin_size = freq * obs_per_bin
        boundaries.append(np.arange(start, end, bin_size))
    boundaries = np.concatenate(boundaries)
    
    return boundaries


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