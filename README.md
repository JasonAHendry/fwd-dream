# fwd-dream
A **forward**-time simulation of malaria transmission and evolution: including **d**rift, **r**ecombination, **e**xtinction, **a**dmixture and **m**eiosis.

## Workflow: Varying Host Prevalence

One application of foward-dream is to look at changes in parasite genetic diversity under different host prevalence regimes. In this experiment, host prevalence can be varied in a variety of ways.

1. Run `notebooks/vary_host_prevalence.ipynb` to vary host prevalence using three different epidemiological parameters.
  - Input: `run_correlation.sh`; contains a base set of parameters from which to vary
  - Output: `run_vary_br.sh`, `run_vary_nv.sh`, `run_vary_gamma.sh`
2. Run `./run_vary_br.sh` to create parameter sets with variable `bite_rate_per_v`.
  - Input: `run_correlation.sh`
  - Calls: `vary_param_set.py`
  - Output: `param_bite_rate_per_v_00.ini`, ... `param_bite_rate_per_v_09.ini`
3. Move to the Rescomp1 cluster.
4. Run `./run_correlations.sh -e 2020-04-10_br-correlation -v bite_rate_per_v -i 100 -s balanced`
  - Tags: `-e`, experiment name; `-v`, parameter being varied; `-i`, number of replicate simulations to run; `-s`, how to seed genomes in simulation
  - Input: `param_bite_rate_per_v_00.ini`, ... `param_bite_rate_per_v_09.ini`
  - Calls: `gen_submit.py`
  - Runs:  `./submit_simulation.sh`; **NB: this will immediately submit these simulations to the cluster.**
  - Output: in `/results/2020-04_br-correlation`; simulation results for the number of replicates indicated **and** for every parameter file conforming to `param_<bite_rate_per_v>_[\d].ini`
  
 Now you will have run a large number of simulations with different host prevalence values, achieved by varying a particular parameter. The next step is to aggregate and do some downstream analysis with these simulation results.
