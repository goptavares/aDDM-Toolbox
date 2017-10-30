# aDDM Toolbox

This toolbox can be used to perform model fitting and to generate simulations
for the attentional drift-diffusion model (aDDM), as well as for the classic
version of the drift-diffusion model (DDM) without an attentional component.

## Prerequisites

aDDM-Toolbox supports Python 2.7 (and Python 3.6 tentatively -- please report 
any bugs). The following libraries are required:

* deap
* future
* matplotlib
* numpy
* pandas
* scipy

## Installing

```
$ pip install addm_toolbox
```

## Running tests

To make sure everything is working correctly after installation, try (from a
UNIX shell, not the Python interpreter):

```
$ addm_toolbox_tests
```

This should take a while to finish, so maybe go get a cup of tea :)

## Getting started

To get a feel for how the algorithm works, try:

```
$ addm_demo --display-figures
```

You can see all the arguments available for the demo using:

```
$ addm_demo --help
```

Here is a list of useful scripts which can be similarly run from a UNIX shell:

* addm_demo
* ddm_pta_test
* addm_pta_test
* addm_pta_mle
* addm_pta_map
* addm_simulate_true_distributions
* addm_basinhopping
* addm_genetic_algorithm
* ddm_mla
* addm_mla

You can also have a look directly at the code in the following modules:

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

## Common issues

If you get errors while using the toolbox under Python 3, try it with
Python 2.7.

If you get a Python RuntimeError with the message "Python is not installed as a 
framework.", try creating the file ~/.matplotlib/matplotlibrc and adding the
following code:

```
backend: TkAgg
```

## Authors

* **Gabriela Tavares** - gtavares@caltech.edu, [goptavares](https://github.com/goptavares)

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE - see the COPYING
file for details.

## Acknowledgments

This toolbox was developed as part of a research project in the [Rangel
Neuroeconomics Lab](http://www.rnl.caltech.edu/) at the California Institute of
Technology.
