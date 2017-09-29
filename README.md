# aDDM Toolbox

This toolbox can be used to perform model fitting and to generate simulations
for the attentional drift-diffusion model (aDDM), as well as for the classic
version of the drift-diffusion model (DDM) without an attentional component.

### Prerequisites

This toolbox requires the following libraries:
* deap
* matplotlib
* numpy
* pandas
* scipy 

### Installing

```
$ pip install addm_toolbox
```

## Running the tests

To make sure everything is working correctly after installation, try:

```
$ addm_run_tests
```

Maybe go get a cup of tea, this should take a while to run :)

## Getting Started

To play with the algorithm, try:

```
$ addm_demo --display_plots
```

You can also have a look at the following modules: 
* demo.py is a script to get started and get a feel for how the algorithm
works.
* addm.py contains the aDDM implementation, with functions to generate model
simulations and obtain the likelihood for a given data trial.
* ddm.py is equivalent to addm.py but for the DDM.
* addm_pta_test.py generates an artificial data set for a given set of aDDM
parameters and attempts to recover these parameters through maximum a
posteriori estimation.
* ddm_pta_test.py is equivalent to addm_pta_test.py but for the DDM.
* addm_pta_mle.py fits the aDDM to a data set by performing maximum
likelihood estimation.
* addm_pta_map.py performs model comparison for the aDDM by obtaining a
posterior distribution over a set of models.
* simulate_addm_true_distributions.py generates aDDM simulations using
empirical data for the fixations.

## Authors

* **Gabriela Tavares** - gtavares@caltech.edu, [goptavares](https://github.com/goptavares)

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE - see the COPYING
file for details.

## Acknowledgments