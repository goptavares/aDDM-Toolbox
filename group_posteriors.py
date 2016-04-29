#!/usr/bin/python

"""
group_posteriors.py
Author: Gabriela Tavares, gtavares@caltech.edu

Posterior distribution estimation for the attentional drift-diffusion model
(aDDM), using a grid search over the 3 free parameters of the model. Data from
all subjects is pooled. aDDM simulations are generated according to the
posterior distribution obtained (instead of generating simulations from a single
model, we sample models from the posterior distribution and simulate them, then
aggregate all simulations).
"""

from multiprocessing import Pool

import argparse
import numpy as np

from addm import (get_trial_likelihood, get_empirical_distributions,
                  generate_probabilistic_simulations)
from util import load_data_from_csv, save_simulations_to_csv


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
                        help="Size of the thread pool.")
    parser.add_argument("--trials-per-subject", type=int, default=100,
                        help="Number of trials from each subject to be used in "
                        "the analysis; if smaller than 1, all trials are used.")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of samples to be drawn from the posterior "
                        "distribution when generating simulations.")
    parser.add_argument("--num-simulations-per-sample", type=int, default=10,
                        help="Number of simulations to be genearated for each "
                        "sample drawn from the posterior distribution.")
    parser.add_argument("--range-d", nargs="+", type=float,
                        default=[0.003, 0.006, 0.009],
                        help="Search range for parameter d.")
    parser.add_argument("--range-sigma", nargs="+", type=float,
                        default=[0.03, 0.06, 0.09],
                        help="Search range for parameter sigma.")
    parser.add_argument("--range-theta", nargs="+", type=float,
                        default=[0.3, 0.5, 0.7],
                        help="Search range for parameter theta.")
    parser.add_argument("--expdata-file-name", type=str, default="expdata.csv",
                        help="Name of experimental data file.")
    parser.add_argument("--fixations-file-name", type=str,
                        default="fixations.csv", help="Name of fixations file.")
    parser.add_argument("--save-simulations", default=False,
                        action="store_true", help="Save simulations to CSV.")
    parser.add_argument("--verbose", default=True, action="store_true",
                        help="Increase output verbosity.")
    args = parser.parse_args()

    pool = Pool(args.num_threads)

    # Load experimental data from CSV file.
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

    # Posteriors estimation for the parameters of the model, using odd trials.
    if args.verbose:
        print("Starting grid search...")
    numModels = (len(args.range_d) * len(args.range_theta) *
                 len(args.range_sigma))
    models = list()
    posteriors = dict()
    for d in args.range_d:
        for theta in args.range_theta:
            for sigma in args.range_sigma:
                model = (d, theta, sigma)
                models.append(model)
                posteriors[model] = 1. / numModels

    subjects = choice.keys()
    for subject in subjects:
        if args.verbose:
            print("Running subject " + subject + "...")
        trials = choice[subject].keys()
        if args.trials_per_subject < 1:
            args.trials_per_subject = len(trials)
        trialSet = np.random.choice(
            [trial for trial in trials if trial % 2],
            args.trials_per_subject, replace=False)
        for trial in trialSet:
            listParams = list()
            for model in models:
                listParams.append(
                    (choice[subject][trial], valueLeft[subject][trial],
                    valueRight[subject][trial], fixItem[subject][trial],
                    fixTime[subject][trial], model[0], model[1], model[2]))
            try:
                likelihoods = pool.map(get_trial_likelihood_wrapper, listParams)
            except Exception as e:
                print("An exception occurred during the likelihood computation "
                      "for subject " + subject + ", trial " + str(trial) +
                      ": " + str(e))
                return

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

        if args.verbose:
            for model in posteriors:
                print("P" + str(model) + " = " + str(posteriors[model]))
            print("Sum: " + str(sum(posteriors.values())))

    if args.verbose:
        print("Finished grid search!")

    # Get empirical distributions from even trials.
    try:
        dists = get_empirical_distributions(
            valueLeft, valueRight, fixItem, fixTime, useOddTrials=False,
            useEvenTrials=True)
    except Exception as e:
        print("An exception occurred while getting empirical distributions: " +
              str(e))
        return
    probLeftFixFirst = dists.probLeftFixFirst
    distLatencies = dists.distLatencies
    distTransitions = dists.distTransitions
    distFixations = dists.distFixations

    # Trial conditions for generating simulations.
    orientations = range(-15,20,5)
    trialConditions = list()
    for oLeft in orientations:
        for oRight in orientations:
            if oLeft != oRight:
                vLeft = np.absolute((np.absolute(oLeft) - 15) / 5)
                vRight = np.absolute((np.absolute(oRight) - 15) / 5)
                trialConditions.append((vLeft, vRight))

    # Generate probabilistic simulations using the posteriors distribution.
    try:
        simul = generate_probabilistic_simulations(
            probLeftFixFirst, distLatencies, distTransitions, distFixations,
            trialConditions, posteriors, args.num_samples,
            args.num_simulations_per_sample)
    except Exception as e:
        print("An exception occurred while running simulations: " + str(e))
        return
    simulRT = simul.RT
    simulChoice = simul.choice
    simulValueLeft = simul.valueLeft
    simulValueRight = simul.valueRight
    simulFixItem = simul.fixItem
    simulFixTime = simul.fixTime
    simulFixRDV = simul.fixRDV

    if args.save_simulations:
        totalTrials = len(simulRT.keys())
        save_simulations_to_csv(
            simulChoice, simulRT, simulValueLeft, simulValueRight, simulFixItem,
            simulFixTime, simulFixRDV, totalTrials)


if __name__ == '__main__':
    main()
