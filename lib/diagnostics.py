import numpy as np
import allel
import scipy.stats
import configparser


# ================================================================= #
# Simulation Diagnostics
#
# ================================================================= #


def calc_derived_params(params):
    """
    Calculate a set of derived parameters `derived_params`
    from base simulation parameters `params`
    """
    derived_params = {}  # not in config, but computed from
    derived_params['lmbda'] = params['bite_rate_per_v']*params['nv']/params['nh']*params['p_inf_h']
    derived_params['psi'] = params['bite_rate_per_v']*params['p_inf_v']
    derived_params['h_v'] = 1/params['gamma']  # human to vector
    derived_params['v_h'] = 1/params['eta']  # vector to human
    return derived_params


def calc_equil_params(params, derived_params):
    """
    Generate a dictionary containing parameters
    needed to compute equilibrium host and vector prevalence
    """
    equil_params = {"lmbda": derived_params['lmbda'], "psi": derived_params['psi'],
                    "gamma": params['gamma'], "eta": params['eta']}
    return equil_params


def calc_x_h(lmbda, psi, gamma, eta):
    """
    Calculate parasite prevalence in hosts
    """
    return (lmbda * psi - gamma * eta) / (lmbda * psi + gamma * psi)


def calc_x_v(lmbda, psi, gamma, eta):
    """
    Calculate parasite prevalence in vectors
    """
    return (lmbda * psi - gamma * eta) / (lmbda * psi + lmbda * eta)


def calc_R0(lmbda, psi, gamma, eta):
    """
    Calculate the R0 statistic given 
    a set of simulation parameters
    """
    return (lmbda * psi) / (gamma * eta)