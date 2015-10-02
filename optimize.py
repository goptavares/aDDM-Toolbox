#!/usr/bin/python

"""
optimize.py
Author: Gabriela Tavares, gtavares@caltech.edu

Maximum likelihood estimation procedure for the attentional drift-diffusion
model (aDDM), using a Basinhopping algorithm to search the parameter space. Data
from all subjects is pooled such that a single set of optimal parameters is
estimated.
"""

from scipy.optimize import basinhopping

import argparse
import numpy as np

from addm import get_trial_likelihood
from util import load_data_from_csv


# Global variables.
choice = dict()
valueLeft = dict()
valueRight = dict()
fixItem = dict()
fixTime = dict()
trialsPerSubject = 0


def get_model_nll(params):
    """
    Computes the negative log likelihood of the global data set given the
    parameters of the aDDM.
    Args:
      params: list containing the 3 model parameters, in the following order: d,
          theta, sigma.
    Returns:
      The negative log likelihood for the global data set and the given model.
    """

    d = params[0]
    theta = params[1]
    sigma = params[2]

    logLikelihood = 0
    subjects = choice.keys()
    for subject in subjects:
        trials = choice[subject].keys()
        trialSet = np.random.choice(trials, trialsPerSubject, replace=False)
        for trial in trialSet:
            try:
                likelihood = get_trial_likelihood(
                    choice[subject][trial], valueLeft[subject][trial],
                    valueRight[subject][trial], fixItem[subject][trial],
                    fixTime[subject][trial], d, theta, sigma=sigma)
            except:
                print("An exception occurred during the likelihood computation "
                      "for subject " + subject + ", trial " + str(trial) + ".")
                raise
            if likelihood != 0:
                logLikelihood += np.log(likelihood)
    print("NLL for " + str(params) + ": " + str(-logLikelihood))
    return -logLikelihood


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials-per-subject", type=int, default=100,
                        help="Number of trials from each subject to be used in "
                        "the analysis; if smaller than 1, all trials are used.")
    parser.add_argument("--num-iterations", type=int, default=100,
                        help="Number of basin hopping iterations.")
    parser.add_argument("--step-size", type=float, default=0.001,
                        help="Step size for use in the random displacement of "
                        "the basin hopping algorithm.")
    parser.add_argument("--initial-d", type=float, default=0.005,
                        help="Initial value for parameter d.")
    parser.add_argument("--initial-theta", type=float, default=0.5,
                        help="Initial value for parameter theta.")
    parser.add_argument("--initial-sigma", type=float, default=0.05,
                        help="Initial value for parameter sigma.")
    parser.add_argument("--lower-bound-d", type=float, default=0.0001,
                        help="Lower search bound for parameter d.")
    parser.add_argument("--upper-bound-d", type=float, default=0.01,
                        help="Upper search bound for parameter d.")
    parser.add_argument("--lower-bound-theta", type=float, default=0,
                        help="Lower search bound for parameter theta.")
    parser.add_argument("--upper-bound-theta", type=float, default=1,
                        help="Upper search bound for parameter theta.")
    parser.add_argument("--lower-bound-sigma", type=float, default=0.001,
                        help="Lower search bound for parameter sigma.")
    parser.add_argument("--upper-bound-sigma", type=float, default=0.1,
                        help="Upper search bound for parameter sigma.")
    parser.add_argument("--expdata-file-name", type=str, default="expdata.csv",
                        help="Name of experimental data file.")
    parser.add_argument("--fixations-file-name", type=str,
                        default="fixations.csv", help="Name of fixations file.")
    args = parser.parse_args()

    global choice
    global valueLeft
    global valueRight
    global fixItem
    global fixTime
    global trialsPerSubject

    # Load experimental data from CSV file and update global variables.
    try:
        data = load_data_from_csv(
            args.expdata_file_name, args.fixations_file_name,
            useAngularDists=True)
    except Exception as e:
        print("An exception occurred while loading the data: " + str(e))
        return
    choice = data.choice
    valueLeft = data.valueLeft
    valueRight = data.valueRight
    fixItem = data.fixItem
    fixTime = data.fixTime

    trialsPerSubject = args.trials_per_subject

    # Initial guess for the parameters: d, theta, sigma.
    initialParams = [args.initial_d, args.initial_theta, args.initial_sigma]

    # Search bounds.
    bounds = [(args.lower_bound_d, args.upper_bound_d),
              (args.lower_bound_theta, args.upper_bound_theta),
              (args.lower_bound_sigma, args.upper_bound_sigma)
             ]

    # Optimize using Basinhopping algorithm.
    minimizerKwargs = dict(method="L-BFGS-B", bounds=bounds)
    try:
        result = basinhopping(
            get_model_nll, initialParams, minimizer_kwargs=minimizerKwargs,
            niter=args.num_iterations,stepsize=args.step_size)
    except Exception as e:
        print("An exception occurred during the basinhopping optimization: " +
              str(e))
        return
    print("Optimization result: " + str(result))


if __name__ == '__main__':
    main()
