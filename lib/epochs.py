import numpy as np
import allel
import scipy.stats
import configparser
from lib.diagnostics import *

# ================================================================= #
# Epochs
#
# ================================================================= #


class Epoch(object):
    """
    Store all information about a single `Epoch`,
    including its start time and duration,
    what parameters are changed,
    and its equilibrium prevalence
    """
    def __init__(self, config, section):
        self.name = section.split("_")[1]
        self.section = section
        self.config = config
        self.occurred = False
        # parameters
        self.adj_keys = config.get(section, "adj_params").split(",")
        self.adj_vals = [float(val) for val in config.get(section, "adj_vals").split(",")]
        self.adj_params = {key: val for key, val in zip(self.adj_keys, self.adj_vals)}
        self.entry_params = {}  # Simulation parameters when epoch is entered
        self.epoch_params = {}  # Simulation parameters at epoch equilibrium
        self.derived_params = {}
        self.equil_params = {}
        self.x_h = None
        self.x_v = None
        # duration
        self.start_time = None
        self.duration = config.get(section, "duration")
        self.time_to_equil = None
        self.t0 = None
        self.telapse = None
        self.t1 = None
        self.gen_rate = None
        self.gens = None
        # approach
        self.approaches = config.get(section, "approach").split(",")
        self.approach_ts = [float(val) for val in config.get(section, "approach_t").split(",")]
        self.approach_inputs = {key: val
                                for key, val
                                in zip(self.adj_keys, list(zip(self.approaches, self.approach_ts)))}
        self.approach_t1 = None
        self.approach_funcs = None
        # sampling (run .set_sampling(), *after* .set_duration())
        self.adj_prev_samp = None
        self.prev_samp_freq = None
        self.prev_samp_t = None
        self.adj_div_samp = None
        self.div_samp_freq = None
        self.div_samp_t = None
        # optionals
        self.calc_genetics = config.getboolean(section, "calc_genetics")
        self.collect_samples = config.getboolean(section, "collect_samples")

    def set_params(self, entry_params):
        """
        Set simulation parameter values for the epoch,
        given entry parameters and adjusted parameters
        """
        self.entry_params = entry_params
        self.epoch_params = entry_params.copy()
        self.epoch_params.update(self.adj_params)

        self.derived_params = calc_derived_params(self.epoch_params)
        self.equil_params = calc_equil_params(self.epoch_params, self.derived_params)
        self.x_h = calc_x_h(**self.equil_params)
        self.x_v = calc_x_v(**self.equil_params)

        approx_ne = self.x_h * self.epoch_params['nh']
        approx_gen_time = (self.derived_params['h_v'] + self.derived_params['v_h'])
        self.time_to_equil = 2 * approx_ne * approx_gen_time

    def set_duration(self, start_time):
        """
        Set the duration of the epoch,
        as well as the number of `generations`,
        the start time and end time
        """
        assert self.time_to_equil is not None, \
            "Time to equilibrium undefined, run `.set_params()` first."
        self.t0 = start_time
        if self.duration == "equil":
            self.telapse = self.time_to_equil
        else:
            self.telapse = int(self.duration)  # we assume an int, specifying duration, has been passed
        self.t1 = self.t0 + self.telapse

        h1 = max([0, self.x_h * self.epoch_params['nh']])
        v1 = max([0, self.x_v * self.epoch_params['nv']])

        self.gen_rate = self.epoch_params['bite_rate_per_v'] * self.epoch_params['nv'] \
                        + self.epoch_params['gamma'] * h1 \
                        + self.epoch_params['eta'] * v1 \
                        + self.epoch_params["migration_rate"]  # include migration
        self.gens = self.telapse * self.gen_rate

    def set_approach(self, n_updates=50.0):
        """
        Set the 'approach' of the epoch,
        i.e. how the adjusted parameters approach
        their new values
        """
        assert self.t0 is not None, \
            "Start time undefined, run `.set_duration()` first."
        self.approach_funcs = {key: self.gen_approach_func(key, approach, approach_t)
                               for (key, (approach, approach_t)) in list(self.approach_inputs.items())}
        self.approach_t1 = self.t0 + np.max(self.approach_ts)
        self.approach_delay = np.max(self.approach_ts) / n_updates

    def gen_approach_func(self, key, approach, approach_t):
        """
        For a given parameter, approach, and approach time,
        return a function specifying the value of the parameter
        as a function of time

        Note that, if key is "nv" or "nh", we return
        and integer value
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
            def approach_func(t, correct=True):
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
            raise Exception("Approach is unrecognized. Must be one of: <step/linear/logistic>.")

        return approach_func

    def approach_params(self, t):
        """
        Return the value of all adjusted parameters
        at time `t`, as a dictionary
        """
        return {key: f(t) for key, f in list(self.approach_funcs.items())}

    def set_sampling(self):
        """
        Set the frequency at which diversity
        and prevalence samples are taken across the Epoch
        """
        # prevalence
        self.adj_prev_samp = self.config.has_option(self.section, "prev_samp_freq")
        if self.adj_prev_samp:
            self.prev_samp_freq = self.config.getfloat(self.section, "prev_samp_freq")
            if self.config.has_option(self.section, "prev_samp_t"):
                self.prev_samp_t = self.config.getint(self.section, "prev_samp_t")
            else:  # defaults to entire epoch
                self.prev_samp_t = self.telapse
        # diversity
        self.adj_div_samp = self.config.has_option(self.section, "div_samp_freq")
        if self.adj_div_samp:
            self.div_samp_freq = self.config.getfloat(self.section, "div_samp_freq")
            if self.config.has_option(self.section, "div_samp_t"):
                self.div_samp_t = self.config.getint(self.section, "div_samp_t")
            else:  # defaults to entire epoch
                self.div_samp_t = self.telapse

    def calc_prev_samps(self, prev_samp_freq):
        """
        Calculate the number of prevalence
        samples that will be collected
        during an epoch
        """
        if self.adj_prev_samp:
            samps_at_adj = self.prev_samp_t / self.prev_samp_freq
            samps_at_base = (self.telapse - self.prev_samp_t) / prev_samp_freq
            n_prev_samps = samps_at_adj + samps_at_base
        else:
            n_prev_samps = self.telapse / prev_samp_freq
        return n_prev_samps

    def calc_div_samps(self, div_samp_freq):
        """
        Calculate the number of diversity
        samples that will be collected
        during an epoch
        """
        if self.adj_div_samp:
            samps_at_adj = self.div_samp_t / self.div_samp_freq
            samps_at_base = (self.telapse - self.div_samp_t) / div_samp_freq
            n_div_samps = samps_at_adj + samps_at_base
        else:
            n_div_samps = self.telapse / div_samp_freq
        return n_div_samps
