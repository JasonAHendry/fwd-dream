<p align="center"><img src="images/logo.png" alt="delve"></p>

A **forward**-time simulation of malaria transmission and evolution: including **d**rift, **r**ecolonisation, **e**xtinction, **a**dmixture and **m**eiosis.

## Install
forward-dream is implemented in python and the dependencies can be installed using [conda](https://docs.conda.io/en/latest/):

```
git clone https://github.com/JasonAHendry/fwd-dream.git
cd fwd-dream
conda env create
conda activate dream
```

## Basic usage

First, activate the conda environment:

```
conda activate dream
```

Then, in the `/fwd-dream` directory, run:

```
python simulation.py -e <expt_name> -p <params/param_set.ini> -s <balanced/random/seed_dir>
```

The `-e` flag specifies your experiment name, e.g. `-e high-transmission`

The `-p` flag the path to your parameter set, which is an `.ini` file in the `/params` directory. In brief, this is how you set all the model parameters for `forward-dream`, and also how you specify different "Epochs" -- i.e. change parameter values *during* a simulation. See `/params/README.txt` for more details.

The `-s` flag specifies how the simulation should be seeded. At present, the simulation hard-coded to seeded with ten infected hosts (see line 328 of `simulation.py`), but the allelic states of the parasites can be `balanced`, `random`, or come from the output of another simulation `seed_dir`.

The simulation will run printing diagnostics to `stdout` in your terminal. Outputs will be deposited in `/results/<expt_name>`. In particular, `op.csv` will contain information about prevalence of hosts and vectors during the simulation, and `og.csv` will contain information about the genetic diversity of the parasite population.


## Workflows

### Simulating variable host prevalence, by different epidemiological causes
One application of foward-dream is to look at differences in parasite genetic diversity at equilibrium, under different host prevalence regimes. Since we are often unsure of what drives host prevalence variation from region to region, we explore varying host prevalence by changing either (i) the vector biting rate, (ii) the vector density, or (iii) the average duration of infection. 

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
 
### Simulating malaria control interventionns
Another application of forward-dream is to explore how genetic diversity statistics behave in non-equilibrium contexts, for example following malaria control interventions. We simulate interventions by changing different simulation parameters, and then follow how a suite of genetic diversity statistics change through time.

1. Create parameter sets that simulate maalria control interventions.
  - I have created three already:
    - `params/param_artemisinin.ini`; vary host duration of infection
    - `params/param_bednets.ini`; vary vector biting rate
    - `params/param_insecticide.ini`; vary vector density
2. Run `gen_submit.py -e 2020-04-10_art-intv -p params/param_artemisinin.ini -i 100 -s balanced`
  - Input: `params/param_artemisinin.ini`
  - Output: `/submit-simulation.sh`; this will contain cluster submission for 100 replicate experiments.
3. Move to Rescomp1 cluster.
4. Run `/submit-simulation.sh`
  - Output: `results/2020-04-10_art-inv` will contain simulation outputs for 100 replicate intervention experiments.

### Simulating migration from a source to a sink population (beta)
I have incorporated a simple migration framework into forward-dream that allows the migration of infections from source population to a sink population. This was designed with the aim to explore genetic diversity statistics in the context of a region with unstable malaria transmission (i.e. R_0 < 1) that receives regular migration from a stable (R_0 > 1) source population. The workflow requires running forward-dream twice: (i) for the source population, and *saving* time-stamped genomes throughout and (ii) for the sink population, specifying the path to the source population simulation's output directory with the `-m` flag, and specifying a migration rate. Below I try to explain in more detail.

1. Create *two* parameter set files, one for the source and one for the sink population.
- I have created two examples:
  - `params/param_migration-high-source.ini`
  - `params/param_migration-high-sink.ini`
- In the source population `.ini`, you should set `store_genomes=True` in the `[Sampling]` section.
  - Genomes will be stored at a frequency specified by `div_samp_freq`
- In the sink population `.ini`, you should set `migration_rate` flag in `[Options]` to your specified rate. Note that you can also initialize the simulation with `migration_rate=0`, and then introduce migration by adjusting the migration rate parameter in an `[Epoch_<name>]` section. I do the later in the example `.ini`.
- The overall initialization (`init_duration`) and epoch durations should be equal for both the `source` and `sink` populations. This keeps the time stamping of the stored genomes aligned with time in the sink population.
- The rate at which migration occurs in should be much less than the rate at which genomes are stored. i.e.:
  - `migration_rate` in the sink `.ini` <<< than `div_samp_freq` in the source `.ini`
  - Why? This helps ensure that every migration event samples a *unique* genome from the source population. If migration happens at too high a rate, you will artificially introduce IBD in the sink population by migrating over identical genomes from the source.
2. Move the parameter files to Rescomp1.
2. Run `gen_submit.py -e 2020-05-06_migration-high-source -p params/param_migration-high-source.ini -s balanced -i 10`.
- Input: `params/param_migration-high-source.ini`
- Output: `submit_simulation.sh`
  - This will run ten iterations of the source population, storing genomes in each. These can be used to feed infections into the sink populations.
3. Run './submit_simulation.sh`
- This submits the source simulations to the cluster.
- Output: `results/2020-05-06_migration-high-source` will contain simulation outputs for 10 replicate source experiments.
4. Run `gen_submit.py -e 2020-05-06_migration-high-sink -p params/param_migration-high-sink.ini -s balanced -m 2020-05-06_migration-high-sink`
- Input: `params/param_migration-high-source.ini`
- Output: `submit_simulation.sh`
  - This will generate sink simulations for each of the ten source simulation.
5. Run './submit_simulation.sh' to submit the simulations to Rescomp1.
- Output: `results/2020-05-06_migration-high-source` will contain simulation outputs for 10 replicate sink experiments.


