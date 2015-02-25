#!/usr/bin/python

# individual_posteriors.py
# Author: Gabriela Tavares, gtavares@caltech.edu

from multiprocessing import Pool

import numpy as np

from handle_fixations import (load_data_from_csv, analysis_per_trial,
    get_empirical_distributions)
from posteriors import generate_probabilistic_simulations


def run_analysis_wrapper(params):
    return analysis_per_trial(*params)


def main():
    numThreads = 9
    pool = Pool(numThreads)

    subject = "gel"
    rt = dict()
    choice = dict()
    distLeft = dict()
    distRight = dict()
    fixItem = dict()
    fixTime = dict()

    # Load experimental data from CSV file.
    data = load_data_from_csv("expdata.csv", "fixations.csv")
    rt[subject] = data.rt[subject]
    choice[subject] = data.choice[subject]
    distLeft[subject] = data.distLeft[subject]
    distRight[subject] = data.distRight[subject]
    fixItem[subject] = data.fixItem[subject]
    fixTime[subject] = data.fixTime[subject]

    # Get item values.
    valueLeft = dict()
    valueRight = dict()
    valueLeft[subject] = dict()
    valueRight[subject] = dict()
    trials = distLeft[subject].keys()
    for trial in trials:
        valueLeft[subject][trial] = np.absolute((np.absolute(
            distLeft[subject][trial])-15)/5)
        valueRight[subject][trial] = np.absolute((np.absolute(
            distRight[subject][trial])-15)/5)

    print("Starting grid search for subject " + subject + "...")
    rangeD = [0.004, 0.0045, 0.005]
    rangeTheta = [0.3, 0.35, 0.4]
    rangeStd = [0.08, 0.085, 0.09]
    numModels = len(rangeD) * len(rangeTheta) * len(rangeStd)

    models = list()
    posteriors = dict()
    for d in rangeD:
        for theta in rangeTheta:
            for std in rangeStd:
                model = (d, theta, std)
                models.append(model)
                posteriors[model] = 1./ numModels

    subjects = rt.keys()
    for subject in subjects:
        trials = rt[subject].keys()
        for trial in trials:
            listParams = list()
            for model in models:
                listParams.append((rt[subject][trial], choice[subject][trial],
                    valueLeft[subject][trial], valueRight[subject][trial],
                    fixItem[subject][trial], fixTime[subject][trial], model[0],
                    model[1], model[2]))
            likelihoods = pool.map(run_analysis_wrapper, listParams)

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
                if likelihoods[i] != 0:
                    prior = posteriors[model]
                    posteriors[model] = likelihoods[i] * prior / denominator
                i += 1

        for model in posteriors:
            print("P" + str(model) + " = " + str(posteriors[model]))
        print("Sum: " + str(sum(posteriors.values())))

    # Get empirical distributions for the data.
    dists = get_empirical_distributions(rt, choice, distLeft, distRight,
        fixItem, fixTime, useOddTrials=True, useEvenTrials=True)
    probLeftFixFirst = dists.probLeftFixFirst
    distTransition = dists.distTransition
    distFirstFix = dists.distFirstFix
    distSecondFix = dists.distSecondFix
    distThirdFix = dists.distThirdFix
    distOtherFix = dists.distOtherFix

    generate_probabilistic_simulations(probLeftFixFirst, distTransition,
        distFirstFix, distSecondFix, distThirdFix, distOtherFix, posteriors,
        numSamples=32, numSimulationsPerSample=1)


if __name__ == '__main__':
    main()
