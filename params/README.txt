`forward-dream` PARAMETER README
--------------------------------
JHendry, 2020/06/01

In this README we describe how to set a param_<set>.ini file for `forward-dream`. 
All parameter values, the structure of Epochs,  and the sampling behaviour 
(i.e. the rate and type of prevalence and genetic data collected) are specified 
by a `param_<set>.ini` file.

When `forward-dream` is run, the parameter file is indicated with the `-p` flag:

python simulation.py -e <expt_name> -p <param_set.ini> -s <balanced/clonal/random>

Individual sections and parameters are described below:


[Demography]
nh = 500  # Number of hosts
nv = 2500  # Number of vectors
nph = 10  # K^h_{max}, number of subcompartments in a host
npv = 10  # K^v_{max}, number of subcompartments in a vector

[Genome]
nsnps = 1000  # Number of SNPs in the genome

[Transmission]
bite_rate_per_v = 0.25  # Daily vector biting rate
p_inf_h = 0.1  # Probability a host is infected, when bitten by an infected vector
gamma = 0.008929  # Host infection clearance rate
p_inf_v = 0.1  # Probability a vector is infected, when biting by an infected host 
eta = 0.2  # Vector clearance rate
p_k_h = 0.1  # Probability an individual subcompartment within a host is transmitted during a bite
p_k_v = 0.1  # Probability an individual subcompartment within a vector is transmitted during a bite

[Evolution]
drift_rate_v = 1  # Drift rate in vectors
drift_rate_h = 1  # Drift rate in hosts
theta_v = 0.0001  # Mutation rate in vectors
theta_h = 0.0001  # Mutation raate in hosts


The [Sampling] section describes how frequently prevelance and genetic data is sampled
during a simulation. In addition, it specifies which genetic diversity statistics should
be calculated during sampling.


[Sampling]
prev_samp_rate = variable  # Rate (in terms of number of events) at which prevalence data is collected
prev_samp_freq = 1000  # Frequency (in terms of days) at which prevalence data is collected
div_samp_rate = variable  # As above, but for genetic data
div_samp_freq = 1000  # As above, but for genetic data
report_rate = 1000000  # Rate (in events) at which prevalence is printed to stdout
track_allele_freqs = False  # Calculate allele frequencies when collecting genetic data?
track_ibd = True  # Calculate IBD when collecting genetic data?
track_sfs = False  # Calculate SFS when collecting genetic data?
track_r2 = False  # Calculate pairwise r2 when collecting genetic data? Note, this is expensive.
track_params = True  # Keep track of parameter values?
store_genomes = False  # Store genomes (used for migration module, in development)


Note that the `max_samples` flag gives number of samples that should be collected when
genetic data is being collected. It is possible there will *not* be this many infected
hosts, in which cases all infected hosts are collected.


[Options]
back_mutation = True  # Should mutation be reversible?
max_samples = 20  # How many samples should you try and collect, when collecting genetic data?
detection_threshold = 0.05  # Minimum fraction sites different for strains to be detected by sequencing
init_duration = 285600  # Duration (in days) of initialization epoch


Following the [Options] section, [Epoch_<name>] sections can be (optionally) included. The [Epoch_<name] sections
all for changes to be made to parameter values *during* a simulation run, to simulate interventions,
seasons, &c. Note they will occur in the *order they are listed*.


[Epoch_ExampleDecreaseNV]
duration = 3650  # Duration of the epoch
adj_params = nv  # Parameter(s) whose value(s) should be adjusted. Note, if >1, give as a python list, e.g. [eta, gamma, ...]
adj_vals = 1000  # Value(s) to which parameter(s) should be changed to. Again if multiple, [0.2, 0.001, ...]
approach = logisitic  # Function describing how the parameter moves to it's new value
approach_t = 200  # Number of days over which change from old to new value occrs
prev_samp_freq = 30  # Optionally, you can adjust the rate at which prevalence...
div_samp_freq = 30  # Or genetic data is collected
calc_genetics = False  # If True, a folder is created at the end of the Epoch storing a larger set of diversity statistics
collect_samples = False  # If True, the genetic material of samples is stored at the end of the Epoch

