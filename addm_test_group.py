#!/usr/bin/python

"""
addm_test_group.py
Author: Gabriela Tavares, gtavares@caltech.edu

Test to check the validity of the addm parameter estimation. Artificil data is
generated using specific parameters for the model. Fixations are sampled from
the data pooled from all subjects. The parameters used for data generation are
then recovered through a posterior distribution estimation procedure.
"""

from multiprocessing import Pool

import argparse
import csv
import numpy as np

from addm import (get_trial_likelihood, get_empirical_distributions,
                  run_simulations)
from util import load_data_from_csv


def get_trial_likelihood_wrapper(params):
    """
    Wrapper for addm.get_trial_likelihood() which takes a single argument.
    Intended for parallel computation using a thread pool.
    Args:
      params: tuple consisting of all arguments required by
          addm.get_trial_likelihood().
    Returns:
      The output of addm.get_trial_likelihood().
    """

    return get_trial_likelihood(*params)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-threads", type=int, default=9,
                        help="size of the thread pool")
    parser.add_argument("--num-trials", type=int, default=800,
                        help="number of artificial data trials to be generated "
                        "per trial condition")
    parser.add_argument("--verbose", default=False, action="store_true",
                        help="increase output verbosity")
    args = parser.parse_args()

    pool = Pool(args.num_threads)

    # Load experimental data from CSV file.
    data = load_data_from_csv("expdata.csv", "fixations.csv",
                              useAngularDists=True)
    valueLeft = data.valueLeft
    valueRight = data.valueRight
    fixItem = data.fixItem
    fixTime = data.fixTime

    # Get empirical distributions.
    dists = get_empirical_distributions(valueLeft, valueRight, fixItem, fixTime)
    probLeftFixFirst = dists.probLeftFixFirst
    distLatencies = dists.distLatencies
    distTransitions = dists.distTransitions
    distFixations = dists.distFixations

    # Parameters for artificial data generation.
    numTrials = 800
    d = 0.006
    theta = 0.5
    sigma = 0.08

    orientations = range(-15,20,5)
    trialConditions = list()
    for oLeft in orientations:
        for oRight in orientations:
            if oLeft != oRight:
                vLeft = np.absolute((np.absolute(oLeft) - 15) / 5)
                vRight = np.absolute((np.absolute(oRight) - 15) / 5)
                trialConditions.append((vLeft, vRight))

    # Generate artificial data.
    if args.verbose:
        print("Running simulations...")
    simul = run_simulations(
        probLeftFixFirst, distLatencies, distTransitions, distFixations,
        args.num_trials, trialConditions, d, theta, sigma=sigma)
    simulChoice = simul.choice
    simulValueLeft = simul.valueLeft
    simulValueRight = simul.valueRight
    simulFixItem = simul.fixItem
    simulFixTime = simul.fixTime

    # Grid search to recover the parameters.
    if args.verbose:
        print("Starting grid search...")
    rangeD = [0.005, 0.006, 0.007]
    rangeTheta = [0.4, 0.5, 0.6]
    rangeSigma = [0.07, 0.08, 0.09]
    numModels = len(rangeD) * len(rangeTheta) * len(rangeSigma)

    models = list()
    posteriors = dict()
    for d in rangeD:
        for theta in rangeTheta:
            for sigma in rangeSigma:
                model = (d, theta, sigma)
                models.append(model)
                posteriors[model] = 1./ numModels

    trials = simulChoice.keys()
    for trial in trials:
        listParams = list()
        for model in models:
            listParams.append(
                (simulChoice[trial], simulValueLeft[trial],
                simulValueRight[trial], simulFixItem[trial],
                simulFixTime[trial], model[0], model[1], model[2]))
        likelihoods = pool.map(get_trial_likelihood_wrapper, listParams)

        # Get the denominator for normalizing the posteriors.
        i = 0
        denominator = 0
        for model in models:
            denominator += posteriors[model] * likelihoods[i]
            i += 1
        if denominator == 0:
            continue

        # Calculate the posteriors after this trial.
        i = 0
        for model in models:
            prior = posteriors[model]
            posteriors[model] = likelihoods[i] * prior / denominator
            i += 1

        if args.verbose and trial % 200 == 0:
            for model in posteriors:
                print("P" + str(model) + " = " + str(posteriors[model]))
            print("Sum: " + str(sum(posteriors.values())))
 
    if args.verbose:
        for model in posteriors:
            print("P" + str(model) + " = " + str(posteriors[model]))
        print("Sum: " + str(sum(posteriors.values())))


if __name__ == '__main__':
    main()
