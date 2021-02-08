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
python simulation.py -e <expt_name> -p <params/param_set.ini>
```

The `-e` flag specifies your experiment name, e.g. `-e high-transmission`.

The `-p` flag gives the path to your parameter set, which is an `.ini` file in the `/params` directory. In brief, this is how you set all the model parameters for `forward-dream`, and also how you specify different "Epochs" -- i.e. change parameter values *during* a simulation. See `/params/README.txt` for more details.

The simulation will run printing diagnostics to `stdout` in your terminal. Outputs will be deposited in `/results/<expt_name>`. In particular, `op.csv` will contain information about prevalence of hosts and vectors during the simulation, and `og.csv` will contain information about the genetic diversity of the parasite population.

### A small example

I've included an example parameter set (`params/param_default-example.ini`) and notebook (`notebooks/sec0_plot-default-example.ipynb`) for you to get started with `forward-dream`. The parameters are set such that the simulation runs with default parameters, but for only 20 years, such that you can run the simulation in a couple minutes on your local machine. To run it, navigate to the `/fwd-dream` directory and run:

```
conda activate dream
python simulation.py -e default -p params/param_default-example.ini
```

The simulation should now print some diagnostics to `stdout`. Once the simulation is done, you should be able to run the notebook `notebooks/sec0_plot-default-example.ipynb` to produce plots that recreate Figure 2 of the `forward-dream` manuscript:

![fig-genetics](figs/sec0_default-genetics.png)

Note that twenty years is not enough time for the population to reach genetic equilibrium. 


## Workflows

Below I describe the workflows I used to run the experiments described in the `forward-dream` manuscript. The workflows are specific to the [BMRC computer cluster at the University of Oxford](https://www.medsci.ox.ac.uk/divisional-services/support-services-1/bmrc/cluster-login). If you plan on re-running these experiments you will likely have to tailor these workflows to your specific cluster setup.

### Simulating variable host prevalence, by different epidemiological causes
One application of foward-dream is to look at differences in parasite genetic diversity at equilibrium, under different host prevalence regimes. Since we are often unsure of what drives host prevalence variation from region to region, we explore varying host prevalence by changing either (i) the vector biting rate, (ii) the vector density, or (iii) the average duration of infection. 

1. `ssh` onto the BMRC cluster.
2. Create a base parameter set in the `params` directory.
- Except for the parameter that is modulated to drive variation in prevalence, the parameters in this file will be held fixed across experiments.
- The `Epochs` in this file will also be held fixed across experiments.
3. Run `gen_correlation-experiment.py -e <expt_name> -p <params/param_file.ini> -n <n_reps>`, setting the `-p` flag to your base parameter set.
- This will generate three files: `submit_correlation-nv.sh`, `submit_correlation-gamma.sh`, and `submit_correlation-br.sh`
4. Source the submission files to submit jobs to the cluster, e.g. `./submit_intervention-nv.sh`.

 
### Simulating malaria control interventionns
Another application of forward-dream is to explore how genetic diversity statistics behave in non-equilibrium contexts, for example following malaria control interventions. We simulate interventions by changing different simulation parameters, and then follow how a suite of genetic diversity statistics change through time.

1. `ssh` onto the BMRC cluster.
2. Create a base parameter set for your intervention experiment in the `params` directory.
- Except for the parameter that is modulated to drive variation in prevalence, the parameters in this file will be held fixed across experiments.
- The `Epochs` in this file will also be held fixed across experiments.
3. Run `gen_intervention-experiment.py  -e <expt_name> -p <params/param_file.ini> -n <n_reps>`, setting the `-p` flag to your base parameter set.
- This will generate three new parameter files: `params_insecticide.ini`, `params_artemisinin.ini`, and `params_bednets.ini`
- It will also generate three submission files: `submit_intervention-nv.sh`, `submit_intervention-gamma.sh`, and `submit_intervention-br.sh` 
4. Source the submission files to submit jobs to the cluster, e.g. `./submit_intervention-nv.sh`.

