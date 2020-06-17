import os
import sys
import getopt
from datetime import datetime
import stat
import json
import numpy as np
import pandas as pd
from collections import Counter
from collections import OrderedDict

from lib.diagnostics import *




print("=" * 80)
print("Generating collection of `.ini` files to run both")
print("correlation and intervention experiments")
print("-" * 80)
print("Command: %s" % " ".join(sys.argv))
print("Run on host: %s" % os.uname().nodename)
print("Operating system: %s" % os.uname().sysname)
print("Machine: %s" % os.uname().machine)
print("Started at: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 80)



# PARSE CLI
print("Parsing user inputs...")
try:
    opts, args = getopt.getopt(sys.argv[1:], "n:c:i:") # [(opt, arg), (opt, arg), (opt, arg)]
except getopt.GetoptError:
    print("Option Error.")

for opt, value in opts:
    if opt == "-n":
        params_name = value
        output_dir = os.path.join("params", "storage", params_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print("Parameter Collection Name:", params_name)
        print("Output Directory:", output_dir)
    elif opt == "-c":
        base_corr_path = value
        os.system("cp %s %s" % (base_corr_path, os.path.join(output_dir, os.path.basename(base_corr_path))))
        print("Base Correlation Experiment Parameter Path:", base_corr_path)
    elif opt == "-i":
        base_intv_path = value
        os.system("cp %s %s" % (base_intv_path, os.path.join(output_dir, os.path.basename(base_intv_path))))
        print("Base Intervention Experiment Parameter Path:", base_intv_path)
    else:
        print("Parameter %s not recognized." % opt)
        sys.exit(2)
print("Done.")
print("")


# SECTION A. Preparing parameter files for correlation experiments
print("-"*80)
# ASSIGN SIMULATION PARAMETERS
config = configparser.ConfigParser()
config.read(base_corr_path)
# Basic
params = {}
demography = {param: int(val) for param, val in config.items('Demography')}
transmission = {param: float(val) for param, val in config.items('Transmission')}
genome = {param: int(val) for param, val in config.items('Genome')}
evolution = {param: float(val) for param, val in config.items('Evolution')}
params.update(demography)
params.update(transmission)
params.update(genome)
params.update(evolution)
# Derived & Equilibrium
derived_params = calc_derived_params(params)
equil_params = calc_equil_params(params, derived_params)
x_h = calc_x_h(**equil_params)
x_v = calc_x_v(**equil_params)
R0 = calc_R0(**equil_params)
bp_per_cM = 2 * params["nsnps"] / 100  # scaled for 1 CO per bivalent
# Options
options = {}
options['back_mutation'] = config.getboolean('Options', 'back_mutation')
options['max_samples'] = eval(config.get('Options', 'max_samples'))
options['detection_threshold'] = eval(config.get('Options', 'detection_threshold'))

print("Correlation Experiment Base Parameter Values:")
print("  No. of Humans:", params['nh'])
print("  No. of Vectors:", params['nv'])
print("  No. of Parasites per Human:", params['nph'])
print("  No. of Parasites per Vector:", params['npv'])
print("  No. Sites in Parasite Genome:", params['nsnps'])
print("  Bite Rate per Vector:", params['bite_rate_per_v'])
print("  Prob. Human Infected per Bite:", params['p_inf_h'])
print("  Prob. Vector Infected per Bite:", params['p_inf_v'])
print("  Prob. Parasite Strain Transmitted to Human:", params['p_k_h'])
print("  Prob. Parasite Strain Transmitted to Vector:", params['p_k_h'])
print("  Number of Oocysts Draw From Geometric with p =", params['p_oocysts'])
print("  Drift Rate in Humans:", params['drift_rate_h'])
print("  Drift Rate in Vectors:", params['drift_rate_v'])
print("  Mutation Rate in Humans (theta):", params['theta_h'])
print("  Mutation Rate in Vectors (theta):", params['theta_v'])
print("  bp per cM [Scaled for 1 CO / bivalent]:", bp_per_cM)
print("  Rate of Human Clearance (gamma):", params['gamma'])
print("  Rate of Vector Clearance (eta):", params['eta'])
print("  Lambda:", derived_params['lmbda'])
print("  Psi:", derived_params['psi'])
print("  Expected Human Prevalence:", x_h)
print("  Expected Vector Prevalence:", x_v)
print("  R0:", R0)
print("    Run with back mutation?", options['back_mutation'])
print("    Limit number of samples collected to:", options['max_samples'])
print("    Set the detection threshold for mixed infections to:", options['detection_threshold'])
print("Done.")
print("")


# SET PREVALENCE RANGE
prevs = np.arange(0.1, 0.9, 0.1)
print("Prevalence Range:", prevs)
print("")




# VARY PREVALENCE: BY NUMBER OF VECTORS
orig_nv = params["nv"]

# Number of vectors for a given prevalence
def nv_at_x_h(x_h, params):
    """
    Solve for the number of vectors `nv` that will
    give a desired host prevalence `x_h`
    given a parameter set.
    """
    a = params["gamma"]*params["eta"]
    b = x_h*params["gamma"]*params["bite_rate_per_v"]*params["p_inf_v"]
    c = (params["bite_rate_per_v"]**2)*params["p_inf_h"]*params["p_inf_v"]*(1 - x_h)
    return params["nh"]*(a + b)/c

# Calculate `nv` matching `x_h`
nvs = nv_at_x_h(prevs, params).astype("int")
nv_dt = {
    "prevs": [],
    "nv": [],
    "x_h": [],
    "x_v": [],
    "t_equil": []
}
for prev, nv in zip(prevs, nvs):
    # Update parameters
    params.update({"nv": nv})
    derived_params = calc_derived_params(params)
    equil_params = calc_equil_params(params, derived_params)
    # Recalculate prevalences
    x_h = calc_x_h(**equil_params) 
    x_v = calc_x_v(**equil_params)
    # Calculate estimated time to equilibrium
    t_equil = 2*x_h*params['nh']*(derived_params['h_v'] + derived_params['v_h'])  # 2*Ne*g
    # Store
    nv_dt["prevs"].append(prev)
    nv_dt["nv"].append(nv)
    nv_dt["x_h"].append(x_h)
    nv_dt["x_v"].append(x_v)
    nv_dt["t_equil"].append(t_equil)  

# Convert to df
nv_df = pd.DataFrame(nv_dt)
column_ord = ["prevs", "nv", "x_h", "x_v", "t_equil"]
nv_df = nv_df[column_ord]

# Revert to original parameter values
print("Reseting default parameters")
params.update({"nv" : orig_nv})
derived_params = calc_derived_params(params)
equil_params = calc_equil_params(params, derived_params)
print("Done.")
print("")




# VARY PREVALENCE: BY HOST CLEARANCE RATE (GAMMA)
orig_gamma = params["gamma"]

def gamma_at_x_h(x_h, params, derived_params):
    """
    Solve for `gamma` that will give a
    desired `x_h` given a set of derived parameters
    """
    a = derived_params['lmbda']*derived_params['psi']*(1 - x_h)
    b = derived_params['psi']*x_h + params['eta']
    return a/b


# Calculate `gamma` matching `x_h`
gammas = gamma_at_x_h(prevs, params, derived_params)
gamma_dt = {
    "prevs": [],
    "gamma": [],
    "x_h": [],
    "x_v": [],
    "t_equil": []
}
for prev, gamma in zip(prevs, gammas):
    # Update parameters
    params.update({"gamma": gamma})
    derived_params = calc_derived_params(params)
    equil_params = calc_equil_params(params, derived_params)
    # Recalculate prevalences
    x_h = calc_x_h(**equil_params) 
    x_v = calc_x_v(**equil_params)
    # Calculate estimated time to equilibrium
    t_equil = 2*x_h*params['nh']*(derived_params['h_v'] + derived_params['v_h'])  # 2*Ne*g
    # Store
    gamma_dt["prevs"].append(prev)
    gamma_dt["gamma"].append(gamma)
    gamma_dt["x_h"].append(x_h)
    gamma_dt["x_v"].append(x_v)
    gamma_dt["t_equil"].append(t_equil)

# Convert to df
gamma_df = pd.DataFrame(gamma_dt)
column_ord = ["prevs", "gamma", "x_h", "x_v", "t_equil"]
gamma_df = gamma_df[column_ord]

# Reset parameters
params.update({"gamma" : orig_gamma})
derived_params = calc_derived_params(params)
equil_params = calc_equil_params(params, derived_params)





# VARY PREVALENCE: BY BITING RATE (BR)
orig_br = params["bite_rate_per_v"]

def bite_rate_per_v_at_x_h(x_h, params, derived_params):
    """
    Solve for `bite_rate_per_v` that will give a
    desired `x_h` given a set of derived parameters
    
    Requires solving a quadratic equation
    
    """
    
    a = (params["nv"]/params["nh"])*params["p_inf_h"]*params["p_inf_v"]*(x_h - 1)
    b = x_h*params["gamma"]*params["p_inf_v"]
    c = params["gamma"]*params["eta"]
    
    num = -b - np.sqrt(b**2 - 4*a*c)
    denom = 2*a
    return num/denom


# Calculate `bite_rate_per_v` matching `x_h`
brs = bite_rate_per_v_at_x_h(prevs, params, derived_params)
br_dt = {
    "prevs": [],
    "br": [],
    "x_h": [],
    "x_v": [],
    "t_equil": []
}
for prev, br in zip(prevs, brs):
    # Update parameters
    params.update({"bite_rate_per_v": br})
    derived_params = calc_derived_params(params)
    equil_params = calc_equil_params(params, derived_params)
    # Recalculate prevalences
    x_h = calc_x_h(**equil_params) 
    x_v = calc_x_v(**equil_params)
    # Calculate estimated time to equilibrium
    t_equil = 2*x_h*params['nh']*(derived_params['h_v'] + derived_params['v_h'])  # 2*Ne*g
    # Store
    br_dt["prevs"].append(prev)
    br_dt["br"].append(br)
    br_dt["x_h"].append(x_h)
    br_dt["x_v"].append(x_v)
    br_dt["t_equil"].append(t_equil)  

# Store in df
br_df = pd.DataFrame(br_dt)
column_ord = ["prevs", "br", "x_h", "x_v", "t_equil"]
br_df = br_df[column_ord]

# Reset
params.update({"bite_rate_per_v" : orig_br})
derived_params = calc_derived_params(params)
equil_params = calc_equil_params(params, derived_params)



# COMPUTE MAXIMUM TIME TO EQUILIBRIUM ACROSS ALL THREE MODALITIES
nv_t_max = nv_df.t_equil.max()
br_t_max = br_df.t_equil.max()
gamma_t_max = gamma_df.t_equil.max()
t_max = max([nv_t_max, br_t_max, gamma_t_max])
print("Max time to equilibrium")
print("nv:", nv_t_max)
print("br:", br_t_max)
print("gamma:", gamma_t_max)
print("")
print("Max. across all:", t_max)
print("               : %.02f (decades)" % (t_max / (365*10)))
print("")

# Set initial duration based on maximum time
# NB: Modifying base file
config.set("Options", "init_duration", str(int(t_max)))
with open(base_corr_path, "w") as config_file:
    config.write(config_file)
    

                  
                  
# WRITE ALL DATA FRAMES
print("Writing data frames...")
nv_df.to_csv(os.path.join(output_dir, "nv_df.csv"), index=False)
br_df.to_csv(os.path.join(output_dir, "br_df.csv"), index=False)
gamma_df.to_csv(os.path.join(output_dir, "gamma_df.csv"), index=False)
print("Done.")
print("")
      
    
    
    
# WRITE ALL PARAMETER FILES
# Number of vectors
print("Modifying Number of Vectors...")
config = configparser.ConfigParser()
config.read(base_corr_path)
section = [s for s in config.sections() if config.has_option(s, "nv")][0]
for val_ix, val in enumerate(nv_df["nv"]):
    config.set(section, "nv", str(val))  # CHANGE parameter value
    val_name = "%s%02d" % ("nv", val_ix)  # this is preferable to the value, as it may be a float
    file_name = "param_" + val_name + ".ini"
    print("   %s \t %s = %s \t HX = %.03f" % (file_name, "nv", val, nv_df.loc[val_ix, "x_h"]))
    with open(os.path.join(output_dir, file_name), "w") as config_file:
        config.write(config_file)
print("Done.")
print("")

# Host Clearance Rate
print("Modifying Host Clearance Rate...")
config = configparser.ConfigParser()
config.read(base_corr_path)
section = [s for s in config.sections() if config.has_option(s, "gamma")][0]
for val_ix, val in enumerate(gamma_df["gamma"]):
    config.set(section, "gamma", str(val))  # CHANGE parameter value
    val_name = "%s%02d" % ("gamma", val_ix)  # this is preferable to the value, as it may be a float
    file_name = "param_" + val_name + ".ini"
    print("   %s \t %s = %s \t HX = %.03f" % (file_name, "nv", val, gamma_df.loc[val_ix, "x_h"]))
    with open(os.path.join(output_dir, file_name), "w") as config_file:
        config.write(config_file)
print("Done.")
print("")

# Biting Rate
print("Modifying Vector Biting Rate...")
config = configparser.ConfigParser()
config.read(base_corr_path)
section = [s for s in config.sections() if config.has_option(s, "bite_rate_per_v")][0]
for val_ix, val in enumerate(br_df["br"]):
    config.set(section, "bite_rate_per_v", str(val))  # CHANGE parameter value
    val_name = "%s%02d" % ("bite_rate_per_v", val_ix)  # this is preferable to the value, as it may be a float
    file_name = "param_" + val_name + ".ini"
    print("   %s \t %s = %s \t HX = %.03f" % (file_name, "bite_rate_per_v", val, br_df.loc[val_ix, "x_h"]))
    with open(os.path.join(output_dir, file_name), "w") as config_file:
        config.write(config_file)
print("Done.")
print("")



# SECTION B. Preparing parameter files for intervention experiments
print("-"*80)
# ASSIGN SIMULATION PARAMETERS
config = configparser.ConfigParser()
config.read(base_intv_path)
# Basic
params = {}
demography = {param: int(val) for param, val in config.items('Demography')}
transmission = {param: float(val) for param, val in config.items('Transmission')}
genome = {param: int(val) for param, val in config.items('Genome')}
evolution = {param: float(val) for param, val in config.items('Evolution')}
params.update(demography)
params.update(transmission)
params.update(genome)
params.update(evolution)
# Derived & Equilibrium
derived_params = calc_derived_params(params)
equil_params = calc_equil_params(params, derived_params)
x_h = calc_x_h(**equil_params)
x_v = calc_x_v(**equil_params)
R0 = calc_R0(**equil_params)
bp_per_cM = 2 * params["nsnps"] / 100  # scaled for 1 CO per bivalent
# Options
options = {}
options['back_mutation'] = config.getboolean('Options', 'back_mutation')
options['max_samples'] = eval(config.get('Options', 'max_samples'))
options['detection_threshold'] = eval(config.get('Options', 'detection_threshold'))

print("Intervention Experiment Base Parameter Values:")
print("  No. of Humans:", params['nh'])
print("  No. of Vectors:", params['nv'])
print("  No. of Parasites per Human:", params['nph'])
print("  No. of Parasites per Vector:", params['npv'])
print("  No. Sites in Parasite Genome:", params['nsnps'])
print("  Bite Rate per Vector:", params['bite_rate_per_v'])
print("  Prob. Human Infected per Bite:", params['p_inf_h'])
print("  Prob. Vector Infected per Bite:", params['p_inf_v'])
print("  Prob. Parasite Strain Transmitted to Human:", params['p_k_h'])
print("  Prob. Parasite Strain Transmitted to Vector:", params['p_k_h'])
print("  Number of Oocysts Draw From Geometric with p =", params['p_oocysts'])
print("  Drift Rate in Humans:", params['drift_rate_h'])
print("  Drift Rate in Vectors:", params['drift_rate_v'])
print("  Mutation Rate in Humans (theta):", params['theta_h'])
print("  Mutation Rate in Vectors (theta):", params['theta_v'])
print("  bp per cM [Scaled for 1 CO / bivalent]:", bp_per_cM)
print("  Rate of Human Clearance (gamma):", params['gamma'])
print("  Rate of Vector Clearance (eta):", params['eta'])
print("  Lambda:", derived_params['lmbda'])
print("  Psi:", derived_params['psi'])
print("  Expected Human Prevalence:", x_h)
print("  Expected Vector Prevalence:", x_v)
print("  R0:", R0)
print("    Run with back mutation?", options['back_mutation'])
print("    Limit number of samples collected to:", options['max_samples'])
print("    Set the detection threshold for mixed infections to:", options['detection_threshold'])
print("Done.")
print("")


# New prevalence value
crash_x_h = 0.2
print("Crash Prevalence: %f" % crash_x_h)
print("")


# Compute parameter adjustments required
crash_nv = nv_at_x_h(crash_x_h, params)
crash_gamma = gamma_at_x_h(crash_x_h, params, derived_params)
crash_br = bite_rate_per_v_at_x_h(crash_x_h, params, derived_params)

# The slowest equilibrium will be when human duration is changed
set_crash_duration = False
if set_crash_duration:
    print("Setting Duration of Crash Epoch...")
    crash_params = params.copy()
    crash_params.update({"gamma": crash_gamma})
    crash_derived_params = calc_derived_params(crash_params)
    t_equil = 2*crash_x_h*params['nh']*(crash_derived_params['h_v'] + crash_derived_params['v_h'])
    config.set("Epoch_Crash", "duration", "%d" % t_equil)  # set duration of crash to be equal to longest
    print("  Duration: %f" % t_equil)
    print("Done.")
    print("")
else:
    print("Keeping default Crash Epoch duration...")
    t_equil = config.getfloat("Epoch_Crash", "duration")
    print("  Duration: %f" % t_equil)
    print("Done.")
    print("")

    

# Modify Epochs
# Number of vectors
print("Creating insecticide intervention...")
print("  Initial Number of Vectors: %d" % params["nv"])
print("  Crash Number of Vectors: %d" % crash_nv)
config.set("Epoch_InitVar", "adj_params", "nv")
config.set("Epoch_InitVar", "adj_vals", "%d" % params["nv"])
config.set("Epoch_Crash", "adj_params", "nv")
config.set("Epoch_Crash", "adj_vals", "%d" % crash_nv)
config.set("Epoch_CrashVar", "adj_params", "nv")
config.set("Epoch_CrashVar", "adj_vals", "%d" % crash_nv)
config.set("Epoch_Recovery", "adj_params", "nv")
config.set("Epoch_Recovery", "adj_vals", "%d" % params["nv"])
with open(os.path.join(output_dir, "param_insecticide.ini"), "w") as config_file:
    config.write(config_file)
print("Done.")
print("")

# Host Clearance Rate
print("Creating artemisinin intervention...")
print("  Initial Host Clearance Rate: %f" % params["gamma"])
print("  Crash Host Clearance Rate: %f" % crash_gamma)
config.set("Epoch_InitVar", "adj_params", "gamma")
config.set("Epoch_InitVar", "adj_vals", "%f" % params["gamma"])
config.set("Epoch_Crash", "adj_params", "gamma")
config.set("Epoch_Crash", "adj_vals", "%f" % crash_gamma)
config.set("Epoch_CrashVar", "adj_params", "gamma")
config.set("Epoch_CrashVar", "adj_vals", "%f" % crash_gamma)
config.set("Epoch_Recovery", "adj_params", "gamma")
config.set("Epoch_Recovery", "adj_vals", "%f" % params["gamma"])
with open(os.path.join(output_dir, "param_artemisinin.ini"), "w") as config_file:
    config.write(config_file)
print("Done.")
print("")

# Vector Biting Rate
print("Creating bednet intervention...")
print("  Initial Biting Rate: %f" % params["bite_rate_per_v"])
print("  Crash Biting Rate: %f" % crash_br)
config.set("Epoch_InitVar", "adj_params", "bite_rate_per_v")
config.set("Epoch_InitVar", "adj_vals", "%f" % params["bite_rate_per_v"])
config.set("Epoch_Crash", "adj_params", "bite_rate_per_v")
config.set("Epoch_Crash", "adj_vals", "%f" % crash_br)
config.set("Epoch_CrashVar", "adj_params", "bite_rate_per_v")
config.set("Epoch_CrashVar", "adj_vals", "%f" % crash_br)
config.set("Epoch_Recovery", "adj_params", "bite_rate_per_v")
config.set("Epoch_Recovery", "adj_vals", "%f" % params["bite_rate_per_v"])
with open(os.path.join(output_dir, "param_bednets.ini"), "w") as config_file:
    config.write(config_file)
print("Done.")
print("")


print("-" * 80)
print("Finished at: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 80)