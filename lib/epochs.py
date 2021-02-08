import os
import json
import configparser
import numpy as np
import pandas as pd

import scipy.stats

from lib.diagnostics import *
from lib.generic import *


# ================================================================= #
# Epoch related functions
#
# ================================================================= #


def save_simulation(t0, h_dt, v_dt, t_h, t_v, out_dir):
    """
    Save the current state of the simulation
    
    Parameters
        t0 : float
            Current time of the simulation in days
        h_dt : dict
            keys : int
                Index of host.
            values : ndarray, float32, shape (nph, nsnps)
                Parasite genomes infecting host.
        v_dt : dict
            keys : int
                Index of vector.
            values : ndarray, float32, shape (npv, nsnps)
                Parasite genomes infecting vector.
        t_h : ndarray, float, shape(n_hosts)
            Array giving the last time (in days) that
            a given host's state was updated.
        t_v : ndarray, float, shape(n_vectors)
            Array giving the last time (in days) that
            a given vector's state was updated.
    Returns
        Saves the state of the simulation in six arrays:
        
        h_ixs : ndarray, int, shape (n_inf_hosts, )
            Indexes of the infected hosts, corresponding to
            `t_h`.
        h_genomes : ndarray, float, shape (nph, nsnps, n_inf_hosts)
            The genetic material carried by each infected host. Note
            that hosts are indexed by the last dimension.
        t_h : as in Parameters
        v_ixs : same as h_ixs but for vectors
        v_genomes : same as h_genomes but for vectors
        t_v : as in Parameters
        
        The time is saved as a dictionary `t0.json`.
        
    """
    
    # Save the current time
    json.dump({"t0" : t0}, open(os.path.join(out_dir, "t0.json"), "w"), default=default)
    
    # Save the time since last update
    np.save(os.path.join(out_dir, "t_h.npy"), t_h)
    np.save(os.path.join(out_dir, "t_v.npy"), t_v)
    
    # Save the host state
    h_ixs = []
    h_genomes = []
    for ix, genome in h_dt.items():
        h_ixs.append(ix)
        h_genomes.append(genome)
    h_ixs = np.array(h_ixs)
    h_genomes = np.dstack(h_genomes)
    np.save(os.path.join(out_dir, "h_ixs.npy"), h_ixs)
    np.save(os.path.join(out_dir, "h_genomes.npy"), h_genomes)
    
    # Save the vector state
    v_ixs = []
    v_genomes = []
    for ix, genome in v_dt.items():
        v_ixs.append(ix)
        v_genomes.append(genome)
    v_ixs = np.array(v_ixs)
    v_genomes = np.dstack(v_genomes)
    np.save(os.path.join(out_dir, "v_ixs.npy"), v_ixs)
    np.save(os.path.join(out_dir, "v_genomes.npy"), v_genomes)
    
    return 0


def parse_parameters(config):
    """
    Pass the parameters in an `.ini` file
    specified by a `config` file
    
    TODO
    - Could check that the length of all of these is correct
    
    Parameters
        config : ConfigParser class
            Object that contains all of the simulation
            parameters (configuration values) loaded 
            from the '.ini' file.
    Returns
        params : dict
            Dictionary of all parameter values
            required for the simulation.
    
    """
    
    params = {}
    
    demography = {param: int(val) for param, val in config.items('Demography')}
    transmission = {param: float(val) for param, val in config.items('Transmission')}
    genome = {param: int(val) for param, val in config.items('Genome')}
    evolution = {param: float(val) for param, val in config.items('Evolution')}
    
    params.update(demography)
    params.update(transmission)
    params.update(genome)
    params.update(evolution)
    
    return params


def update_vectors(nv, v, t_v, v_dt):
    """
    Update the current vector population such that
    the number of vectors `nv` and the vector population
    state data structures (`v`, `t_v` and `v_dt`) match
    
    This may involve either killing or creating vectors.
    
    Parameters
        nv: int
            The number of vectors that the simulation *should*
            currently contain; i.e. the number given by the
            parameter file as `params['nv']`. Note this may
            change as the simulation passes through Epochs.
        v: ndarray, int8, shape(n_vectors)
            The infection status of all vectors.
        t_v: ndarray, float32, shape(n_vectors)
            The last time each vectors infection was
            updated.
        v_dt: dict, shape(n_infected_vectors)
            keys: int
                Indices for infected vectors.
            values: ndarray, float32, shape(npv, nsnps)
                Parasite genomes held by infected vectors.
    
    Returns:
        v: ndarray, int8, shape(n_vectors)
            The infection status of all vectors.
        t_v: ndarray, float32, shape(n_vectors)
            The last time each vectors infection was
            updated.
        v_dt: dict, shape(n_infected_vectors)
            keys: int
                Indices for infected vectors.
            values: ndarray, float32, shape(npv, nsnps)
                Parasite genomes held by infected vectors.
    
    """
    
    nv = int(nv)
    if nv > len(v):  # Create vectors
        n_missing_v = nv - len(v)
        v = np.concatenate((v, np.zeros(n_missing_v, dtype='int8')))
        t_v = np.concatenate((t_v, np.zeros(n_missing_v)))
        
    elif nv < len(v):  # Kill vectors
        v = v[:nv]  # Random order, so this is a random subset
        v_dt = {ix: genomes for ix, genomes in v_dt.items() if ix < nv}
        t_v = t_v[:nv]
        
    return v, t_v, v_dt


# ================================================================= #
# class Epoch and Epochs
# 
#
# ================================================================= #


class Epoch(object):
    """
    Store information about a single Epoch
    in fwd-dream    
    
    Example section from an `params_<set>.ini`:
    
    [Epoch_Crash]
    duration = 36500
    adj_params = gamma
    adj_vals = 0.012195
    approach = logistic
    approach_t = 30
    div_samp_freq = 5
    div_samp_t = 365
    prev_samp_freq = 5
    prev_samp_t = 365
    calc_genetics = True
    save_state = True
    
    Example Usage:
    
    epoch = Epoch(config, "Epoch_Crash")
    epoch.set_params(entry_params)
    epoch.set_timings(start_time)
    epoch.set_approach()
    epoch.set_sampling()
    
    
    """
    
    def __init__(self, config, section):
        
        # Epoch name
        self.config = config
        self.section = section
        
        if not section.startswith("Epoch_"):
            raise ValueError("Epochs sections must begin with 'Epoch_'.")
            
        self.name = section.split("_")[1]
        
        # Epoch time        
        self.duration = eval(config.get(section, "duration"))  # evalute to parse 'None'
        self.t0 = None
        self.tdelta = None
        self.t1 = None

        # Epoch entry and equilibrium parameters
        self.begun = False
        self.entry_params = None
        self.epoch_params = None
        self.x_h = None
        self.x_v = None
        
        # Parameter changes entering epoch
        self.adj_keys = [s.strip() for s in config.get(section, "adj_params").split(",")]
        self.adj_vals = [float(val) for val in config.get(section, "adj_vals").split(",")]
        self.adj_params = {key: val for key, val in zip(self.adj_keys, self.adj_vals)}
        
        # Timing of parameter changes
        self.approach = [s.strip() for s in config.get(section, "approach").split(",")]
        self.approach_ts = [float(val) for val in config.get(section, "approach_t").split(",")]
        self.approach_t1 = None  # The last time we will update parameters
        self.tparam = None  # Last time the parameters were updated
        self.param_update_freq = None
        
        if not len(self.adj_params) == len(self.adj_vals):
            raise ValueError("The number of parameters adjusted by `adj_params` must equal" + \
                             "the number of values given by `adj_vals`.")
            
        if not len(self.approach) == len(self.approach_ts):
            raise ValueError("The number of approach functions given by `approach` must equal" + \
                             "the number of approach times given by `approach_ts`.")
        
        # Longitudinal sampling of genetic diversity
        self.adj_div_samp = None
        self.div_samp_freq = None
        self.div_samp_t = None
        
        # Longitudinal sampling of prevalence
        self.adj_prev_samp = None
        self.prev_samp_freq = None
        self.prev_samp_t = None
        
        # Storage
        self.calc_genetics = config.getboolean(section, "calc_genetics")
        self.save_state = config.getboolean(section, "save_state")
        
    
    def set_params(self, entry_params):
        """
        Set entry parameters and equilibrium parameters
        for the Epoch
        
        Parameters
            entry_params: dict
                Simulation parameters upon entry to the
                epoch.
        Returns
            Null
        
        """
        # Set entry parameters
        self.entry_params = entry_params.copy()
        
        # Compute epoch equilbrium paramters
        self.epoch_params = entry_params.copy()
        self.epoch_params.update(self.adj_params)
        
        # Compute epoch host and vector prevalence
        derived_params = calc_derived_params(self.epoch_params)
        equil_params = calc_equil_params(self.epoch_params, derived_params)
        self.x_h = calc_x_h(**equil_params)
        self.x_v = calc_x_v(**equil_params)
        
        
    def set_timings(self, start_time):
        """
        Set the start and end time of the Epoch
        using the `start_time` and duration
        information from `self.duration`
        
        Parameters
            start_time: float
                The start time of the Epoch.
                
        Returns
            Null
        
        """
        if self.entry_params is None:
            raise ValueError("Must run `.set_params()` before running `.set_timings()`.")
        
        # Start time
        self.t0 = start_time
        
        # Duration
        if self.duration is None:
            # Calculate approximate equilibrium time
            derived_params = calc_derived_params(self.epoch_params)
            approx_ne = self.x_h * self.epoch_params["nh"]
            approx_generation_t = derived_params["h_v"] + derived_params["v_h"]
            self.tdelta =  4.21 * approx_ne * approx_generation_t  # Covers TMRCA 95% of time
        else:
            self.tdelta = int(self.duration)  # assuming an int has been passed
        
        # End time
        self.t1 = self.t0 + self.tdelta
    
    
    def set_approach(self, n_updates=50.0):
        """
        Prepare the approach times for the Epoch's parameter 
        changes; the time over which they transition from their 
        entry and adjusted values.
        
        Parameters
            n_updates: int
                The number
        
        Returns
            Null
        
        """
        if self.t0 is None:
            raise ValueError("Must run `.set_timings()` before running `.set_approach()`.")
        
        # Generate a dictionary that holds functions for each parameter to be updated,
        # These functions return the parameter's value at a given time.
        self.approach_funcs = {key: self.gen_approach_func(key, a, a_t)
                               for (key, a, a_t) in zip(self.adj_params,
                                                        self.approach,
                                                        self.approach_ts)}
        
        # We don't continuously update parameters, but at a frequency defined below
        self.approach_t1 = self.t0 + max(self.approach_ts)
        self.param_update_freq = max(self.approach_ts) / n_updates
        
            
    def set_sampling(self):
        """
        Set the sampling rate of prevalence and
        genetic diversity data during the Epoch
        
        Parameters
            Null
        Returns
            Null
        
        """
        if self.t1 is None:
            raise ValueError("Must run `.set_timings()` before running `.set_sampling()`.")
        
        # Prevalence
        self.adj_prev_samp = self.config.has_option(self.section, "prev_samp_freq")
        if self.adj_prev_samp:
            self.prev_samp_freq = self.config.getfloat(self.section, "prev_samp_freq")
            if self.config.has_option(self.section, "prev_samp_t"):
                self.prev_samp_t = self.config.getfloat(self.section, "prev_samp_t")
            else:
                self.prev_samp_t = self.tdelta # until end of epoch
        
        # Diversity
        self.adj_div_samp = self.config.has_option(self.section, "div_samp_freq")
        if self.adj_div_samp:
            self.div_samp_freq = self.config.getfloat(self.section, "div_samp_freq")
            if self.config.has_option(self.section, "div_samp_t"):
                self.div_samp_t = self.config.getfloat(self.section, "div_samp_t")
            else:
                self.div_samp_t = self.tdelta  # until end of epoch
        
    
    def gen_approach_func(self, key, approach, approach_t):
        """
        Generate functions that define gradual updates
        
        Parameters:
            key: str
                Parameter for which we will generate an
                update function.
            approach: str
                The functional form that will be used to
                set the parameter updates. Can be of
                step, linear, logisitic.
            approach_t: float
                The time frame over which the parameter
                will be updated.
        
        Returns
            approach_function: function
                This is a function that, given a time,
                will return the parameter value.

        """
        entry_val = self.entry_params[key]
        epoch_val = self.epoch_params[key]

        if approach == "step":
            def approach_func(t):
                if t <= self.t0 + approach_t/2:
                    val = entry_val
                else:
                    val = epoch_val
                return val if key != ("nv" or "nh") else int(val)

        elif approach == "linear":
            def approach_func(t):
                if t <= self.t0:
                    val = entry_val
                elif t > self.t0 + approach_t:
                    val = epoch_val
                else:
                    b = entry_val
                    m = (epoch_val - entry_val)/approach_t
                    val = b + m * (t - self.t0)
                return val if key != ("nv" or "nh") else int(val)

        elif approach == "logistic":
            def approach_func(t, correct=False):
                mu = self.t0 + approach_t / 2
                n_sds = 10.0
                scale = approach_t / n_sds
                unscaled_func = scipy.stats.logistic(mu, scale)
                val = entry_val + (epoch_val - entry_val) * unscaled_func.cdf(t)
                if correct:  # logistic is asymptotic, linear adjustment to get boundaries exact
                    offset = (epoch_val - entry_val) * unscaled_func.cdf(self.t0)
                    m = 2 * offset / approach_t
                    val += m * (t - mu)
                return val if key != ("nv" or "nh") else int(val)
        else:
            raise ValueError("Approach is unrecognized. Must be one of: <step/linear/logistic>.")

        return approach_func
    
    
    def adjust_params(self, t):
        """
        This method will determine whether or not it's time to 
        update the parameters again
        
        Parameters
            t : float
                Current time in the simulation in days.
        Returns
            _ : bool
                True if parameters should be updated.
        
        """
        if (t - self.t0) >= self.approach_t1:
            return False
        elif (t - self.tparam) >= self.param_update_freq:
            return True
        else:
            return False
        
        
    def get_params(self, t):
        """
        Return the value of all adjusted parameters
        at time `t`, as a dictionary
        
        Parameters
            t : float
                Current time in the simulation in days.
        Returns
            _ : dict
                keys : str, parameter names
                values : appropriate value of parameter at time t
        
        """
        self.tparam = t
        return {key: f(t) for key, f in self.approach_funcs.items()}

     
class Epochs(object):
    """
    Co-ordinate multiple Epoch classes
    
    """
    def __init__(self, params, config):
        
        # Parse
        self.params = params.copy()  # Need to copy, as will change during simulation
        self.config = config
        
        # Define initialisation variables
        self.derived_params = calc_derived_params(self.params)
        self.equil_params = calc_equil_params(self.params, self.derived_params)
        self.init_x_h = calc_x_h(**self.equil_params)
        self.init_x_v = calc_x_v(**self.equil_params)
        
        # Coordinate the Epochs
        self.init_duration = None
        self.epoch_sections = None
        self.exist = None  # Are there any Epochs?
        self.max_t0 = None  # What is the total runtime in days
        self.current = None  # Points to the current epoch

        
    def set_initialisation(self, verbose=False):
        """
        Set the initialisation duration of the simulation
        
        Parameters
            verbose : bool
        Returns
            Null
        """
        
        self.init_duration = eval(self.config.get('Options', 'init_duration'))
        if self.init_duration is None:
            if verbose:
                print("Initialising simulation to approximate equilibrum.")
            ne = self.init_x_h*self.params['nh']
            g = (self.derived_params['h_v'] + self.derived_params['v_h']) 
            time_to_equil = 4.21 * ne * g  # Covers TMRCA 95% of time
            self.init_duration = time_to_equil
        else:
            if verbose:
                print("Initialising simulation to a user-specified duration.")
        if verbose:
            print("  Initialisation duration: %d days = %d years" % (self.init_duration, self.init_duration/365))
        
        
    def prepare_epochs(self, verbose=False):
        """
        With this method we will prepare all of the epochs
             
        Parameters
            verbose : bool
        Returns
            Null
        """
        
        # Collect 'Epoch_' sections
        self.epoch_sections = [s for s in self.config.sections() if "Epoch" in s]
        
        # If Epochs exist, prepare them
        if len(self.epoch_sections) > 0:
            self.exist = True
            self.epochs = [Epoch(self.config, s) for s in self.epoch_sections]
            if verbose: print("Epochs")
            for (i, epoch) in enumerate(self.epochs):
                if i == 0:
                    epoch.set_params(self.params)
                    epoch.set_timings(self.init_duration)  # begins at end of initialization
                    epoch.set_approach()
                    epoch.set_sampling()
                else:
                    epoch.set_params(entry_params=self.epochs[i-1].epoch_params)
                    epoch.set_timings(start_time=self.epochs[i-1].t1)  # begins at end `.t1` of previous epoch
                    epoch.set_approach()
                    epoch.set_sampling()
                if verbose:
                    print(" ", i+1, ":", epoch.name)
                    print("    Begins: %d, Ends: %d" % (epoch.t0, epoch.t1))
                    print("    Duration: %d days = %d years" % (epoch.tdelta, epoch.tdelta/365))
                    print("    Adjusting Parameter(s):", epoch.adj_keys)
                    print("    To Value(s):", epoch.adj_vals)
                    print("    via.:", epoch.approach)
                    print("    Approach Time(s):", epoch.approach_ts)
                    print("    Host Prevalence: %.03f, Vector: %.03f" % (epoch.x_h, epoch.x_v))
                    print("    Adjust Prevalence Sampling:", epoch.adj_prev_samp)
                    if epoch.adj_prev_samp:
                        print("    ...to every %d days for %d days." \
                              % (epoch.prev_samp_freq, epoch.prev_samp_t))
                    print("    Adjust Diversity Sampling:", epoch.adj_div_samp)
                    if epoch.adj_div_samp:
                        print("    ...to every %d days for %d days." \
                              % (epoch.div_samp_freq, epoch.div_samp_t))
            
            self.max_t0 = self.epochs[-1].t1  # the end of the simulation
            if verbose:
                print("  Total Duration: %d days = %d years" % (self.max_t0, self.max_t0/365))
        
    def update_time(self, t):
        """
        Check if current epoch needs to be changed,
        given the time `t`
        
        If we have passed the start time of an Epoch,
        but it has not yet begun, we assign it as
        the current epoch.
        
        """
        
        for epoch in self.epochs:
            if t > epoch.t0 and not epoch.begun:
                self.current = epoch
          
        
    def write_epochs(self, out_dir, verbose=False):
        """
        Write a dataframe `epoch_df.csv`, each row
        of which contains information about and Epoch
        within Epochs
        
        Parameters
            out_dir : str
                Path to output direcftory.
            verbose : bool
                Print to stdout?
        Returns
            Null
        
        """
        
        if self.exist:
            print("Writing Epochs dataframe...")

            epoch_dt = {
                "name": ["init"],
                "t0": [0],
                "t1": [self.init_duration],
                "param": [""],
                "val": [""],
                "x_h": [self.init_x_h],
                "x_v": [self.init_x_v]}
            
            for epoch in self.epochs:
                epoch_dt["name"].append(epoch.name)
                epoch_dt["t0"].append(epoch.t0)
                epoch_dt["t1"].append(epoch.t1)
                epoch_dt["param"].append(epoch.adj_keys)
                epoch_dt["val"].append(epoch.adj_vals)
                epoch_dt["x_h"].append(epoch.x_h)
                epoch_dt["x_v"].append(epoch.x_v)
    
            epoch_df = pd.DataFrame(epoch_dt)
            epoch_df.to_csv(os.path.join(out_dir, "epoch_df.csv"), index=False)
            print("Done.")
            print("")
        else:
            print("No Epochs to write.")

            