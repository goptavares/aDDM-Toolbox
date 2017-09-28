# aDDM Toolbox

This toolbox can be used to perform model fitting and to generate simulations
for the attentional drift-diffusion model (aDDM), as well as a classic version
of the drift-diffusion model (DDM) without an attentional component.

## Getting Started

* demo.py is a script to get started and get a feel for how the algorithm
works.
* addm.py contains the aDDM implementation, with functions to generate model
simulations and obtain the likelihood for a given data trial.
* ddm.py is equivalent to addm.py but for the DDM.
* addm_test.py generates an artificial data set for a given set of aDDM
parameters and attempts to recover these parameters through maximum a
posteriori estimation.
* ddm_test.py is equivalent to addm_test.py but for the DDM.
* addm_mle.py performs fits the aDDM to a data set by performing maximum
likelihood estimation.
* addm_posteriors.py performs model comparison for the aDDM by obtaining a
posterior distribution over a set of models.
* simulate_addm_true_distributions.py generates aDDM simulations using
empirical data for the fixations.

### Prerequisites

```
Examples
```

### Installing

```
Examples
```

## Running the tests

```
Examples
```

## Authors

* **Gabriela Tavares** - gtavares@caltech.edu, [goptavares](https://github.com/goptavares)

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE - see the COPYING
file for details.

## Acknowledgments