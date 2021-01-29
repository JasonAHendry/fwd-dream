#  `forward-dream` PARAMETER README
#  --------------------------------
#  JHendry, 2021/01/29
#
#  In this README we describe how to set a param_<set>.ini file for `forward-dream`. 
#  All parameter values, the structure of Epochs,  and the sampling behaviour 
#  (i.e. the rate and type of prevalence and genetic data collected) are specified 
#  by a `param_<set>.ini` file.
#
#  When `forward-dream` is run, the parameter file is indicated with the `-p` flag:
#
#  python simulation.py -e <expt_name> -p <params/param_set.ini> -s <balanced/clonal/random >
#  Individual sections and parameters are described below:


[Demography]
nh = 500  # Number of hosts
nv = 2500  # Number of vectors
nph = 10  # Number of subcompartments in a host
npv = 10  # Number of subcompartments in a vector

[Genome]
nsnps = 1000  # Number of SNPs in the genome

[Transmission]
bite_rate_per_v = 0.25  # Daily vector biting rate
p_inf_h = 0.1  # Probability a host is infected, when bitten by an infected vector
gamma = 0.005  # Host infection clearance rate
p_inf_v = 0.1  # Probability a vector is infected, when biting by an infected host 
eta = 0.2  # Vector clearance rate
p_oocysts = 0.5  # Number of oocysts distributed like ~Geo(p_oocysts)
p_k_h = 0.1  # Probability an individual subcompartment within a host is transmitted during a bite
p_k_v = 0.1  # Probability an individual subcompartment within a vector is transmitted during a bite

[Evolution]
drift_rate_v = 1  # Moran model, drift rate in vectors
drift_rate_h = 1  # Moran model, drift rate in hosts
theta_v = 0.0001  # Moran model, mutation rate in vectors
theta_h = 0.0001  # Moran model, mutation rate in hosts

[Sampling]
prev_samp_freq = 30  # Frequency (in days) at which prevalence data is collected
div_samp_freq = 30  # As above, but for genetic data
report_rate = 100000  # A report is printed to stdout every `report_rate` events

[Options]
n_seed = 25  # Seed the simulation with `n_seed` infected hosts
binary_genomes = True  # If True, SNPs are in {0, 1} and mutation reversible; False, SNPs are in [0, 1]
max_samples = 20  # How many samples should be collected when computing genetic diversity statistics?
detection_threshold = 0.01  # Minimum fraction sites different for strains to be detected by sequencing
init_duration = 285600  # Initialisation time in days; if None, forward-dream will estimate equilibrium time


#  [Epoch_<name>] sections can be (optionally) included. The [Epoch_<name>] sections
#  allow for changes to be made to parameter values *during* a simulation, to simulate interventions,
#  seasons, &c. Note they will occur in the *order they are listed*.


[Epoch_ExampleDecreaseNV]
duration = 3650  # Duration of the epoch
adj_params = nv  # Parameter(s) whose value(s) should be adjusted. Note, if >1, give as a python list, e.g. [eta, gamma, ...]
adj_vals = 1000  # Value(s) to which parameter(s) should be changed to. Again if multiple, [0.2, 0.001, ...]
approach = logisitic  # Function describing how the parameter moves to its new value
approach_t = 200  # Number of days over which change from old to new value occrs
prev_samp_freq = 30  # Optionally, you can adjust the rate at which prevalence...
div_samp_freq = 30  # Or genetic data is collected
calc_genetics = False  # If True, store genetic diversity statistics entering Epoch
save_state = False  # If True, store the state of the simulation upon Epoch entry

