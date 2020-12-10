import numpy as np
import os
import json
import scipy.stats
import configparser
from lib.diagnostics import *
from lib.generic import *


# ================================================================= #
# Save Simulation State
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
        Null
        
    """
    
    # Save the time
    json.dump({"t0" : t0}, open(os.path.join(out_dir, "t0.json"), "w"), default=default)
    
    # Save infect genomes
    json.dump({int(k): v for k, v in h_dt.items()},  # necessary to make JSON serialisable
              open(os.path.join(out_dir, "h_dt.json"), "w"), 
              default=default)
    json.dump({int(k): v for k, v in v_dt.items()}, 
              open(os.path.join(out_dir, "v_dt.json"), "w"), 
              default=default)
    
    # Save time since last update
    np.save(os.path.join(out_dir, "t_h.npy"), t_h)
    np.save(os.path.join(out_dir, "t_v.npy"), t_v)
    
    return 0


# ================================================================= #
# Parse input parameters
#
# ================================================================= #


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


# ================================================================= #
# Update vector population
#
# ================================================================= #


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
# Define Epoch behaviour
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
        self.duration = config.get(section, "duration")
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
        if self.duration == "equilibrium":
            # Calculate approximate equilibrium time
            derived_params = calc_derived_params(self.epoch_params)
            approx_ne = self.x_h * self.epoch_params["nh"]
            approx_generation_t = derived_params["h_v"] + derived_params["v_h"]
            self.tdelta =  2 * approx_ne * approx_generation_t  # E[TMRCA] = 2Ng
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