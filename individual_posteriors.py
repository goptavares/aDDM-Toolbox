#!/usr/bin/python

"""
individual_posteriors.py
Author: Gabriela Tavares, gtavares@caltech.edu

Posterior distribution estimation for the attentional drift-diffusion model
(aDDM), using a grid search over the 3 free parameters of the model. Data from a
single subject is used. aDDM simulations are generated according to the
posterior distribution obtained (instead of generating simulations from a single
model, we sample models from the posterior distribution and simulate them, then
aggregate all simulations).
"""

from multiprocessing import Pool

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
    numThreads = 9
    pool = Pool(numThreads)

    subject = "cai"
    choice = dict()
    valueLeft = dict()
    valueRight = dict()
    fixItem = dict()
    fixTime = dict()

    # Load experimental data from CSV file.
    data = load_data_from_csv("expdata.csv", "fixations.csv",
                              useAngularDists=True)
    choice[subject] = data.choice[subject]
    valueLeft[subject] = data.valueLeft[subject]
    valueRight[subject] = data.valueRight[subject]
    fixItem[subject] = data.fixItem[subject]
    fixTime[subject] = data.fixTime[subject]

    # Posteriors estimation for the parameters of the model, using odd trials.
    print("Starting grid search for subject " + subject + "...")
    rangeD = [0.004, 0.0045, 0.005]
    rangeTheta = [0.3, 0.35, 0.4]
    rangeSigma = [0.08, 0.085, 0.09]
    numModels = len(rangeD) * len(rangeTheta) * len(rangeSigma)

    models = list()
    posteriors = dict()
    for d in rangeD:
        for theta in rangeTheta:
            for sigma in rangeSigma:
                model = (d, theta, sigma)
                models.append(model)
                posteriors[model] = 1./ numModels

    subjects = choice.keys()
    for subject in subjects:
        trials = choice[subject].keys()
        for trial in trials:
            if not trial % 2:
                continue
            listParams = list()
            for model in models:
                listParams.append(
                    (choice[subject][trial], valueLeft[subject][trial],
                    valueRight[subject][trial], fixItem[subject][trial],
                    fixTime[subject][trial], model[0], model[1], model[2]))
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

        for model in posteriors:
            print("P" + str(model) + " = " + str(posteriors[model]))
        print("Sum: " + str(sum(posteriors.values())))

    # Get empirical distributions from even trials.
    dists = get_empirical_distributions(
        valueLeft, valueRight, fixItem, fixTime, useOddTrials=False,
        useEvenTrials=True)
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
    simul = generate_probabilistic_simulations(
        probLeftFixFirst, distLatencies, distTransitions, distFixations,
        trialConditions, posteriors, numSamples=32, numSimulationsPerSample=1)
    simulRT = simul.RT
    simulChoice = simul.choice
    simulValueLeft = simul.valueLeft
    simulValueRight = simul.valueRight
    simulFixItem = simul.fixItem
    simulFixTime = simul.fixTime
    simulFixRDV = simul.fixRDV

    totalTrials = len(simulRT.keys())
    save_simulations_to_csv(
        simulChoice, simulRT, simulValueLeft, simulValueRight, simulFixItem,
        simulFixTime, simulFixRDV, totalTrials)


if __name__ == '__main__':
    main()
