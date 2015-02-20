#!/usr/bin/python

# group_posteriors.py
# Author: Gabriela Tavares, gtavares@caltech.edu

from multiprocessing import Pool

import numpy as np

from group_fitting import save_simulations_to_csv
from handle_fixations import (load_data_from_csv, analysis_per_trial,
    get_empirical_distributions, run_simulations)


def generate_probabilistic_simulations(probLeftFixFirst, distTransition,
    distFirstFix, distSecondFix, distMiddleFix, posteriors, numSamples=100,
    numSimulationsPerSample=10):
    posteriorsList = list()
    models = dict()
    i = 0
    for model, posterior in posteriors.iteritems():
        posteriorsList.append(posterior)
        models[i] = model
        i += 1

    # Parameters for generating simulations.
    orientations = range(-15,20,5)
    trialConditions = list()
    for oLeft in orientations:
        for oRight in orientations:
            if oLeft != oRight:
                trialConditions.append((oLeft, oRight))

    rt = dict()
    choice = dict()
    valueLeft = dict()
    valueRight = dict()
    fixItem = dict()
    fixTime = dict()

    numModels = len(models.keys())
    trialCount = 0
    for i in xrange(numSamples):
        # Sample model from posteriors distribution.
        modelIndex = np.random.choice(np.array(range(numModels)),
            p=np.array(posteriorsList))
        model = models[modelIndex]
        d = model[0]
        theta = model[1]
        std = model[2]

        # Generate simulations with the sampled model.
        simul = run_simulations(probLeftFixFirst, distTransition, distFirstFix,
            distSecondFix, distMiddleFix, numSimulationsPerSample,
            trialConditions, d, theta, std=std)
        for trial in simul.rt.keys():
            rt[trialCount] = simul.rt[trial]
            choice[trialCount] = simul.choice[trial]
            fixTime[trialCount] = simul.fixTime[trial]
            fixItem[trialCount] = simul.fixItem[trial]
            valueLeft[trialCount] = np.absolute((np.absolute(
                simul.distLeft[trial])-15)/5)
            valueRight[trialCount] = np.absolute((np.absolute(
                simul.distRight[trial])-15)/5)
            trialCount += 1

    numTrials = len(rt.keys())
    save_simulations_to_csv(choice, rt, valueLeft, valueRight, fixItem, fixTime,
        numTrials)


def run_analysis_wrapper(params):
    return analysis_per_trial(*params)


def main():
    trialsPerSubject = 500
    numThreads = 9
    pool = Pool(numThreads)

    # Load experimental data from CSV file.
    data = load_data_from_csv("expdata.csv", "fixations.csv")
    rt = data.rt
    choice = data.choice
    distLeft = data.distLeft
    distRight = data.distRight
    fixItem = data.fixItem
    fixTime = data.fixTime

    # Get item values.
    valueLeft = dict()
    valueRight = dict()
    subjects = distLeft.keys()
    for subject in subjects:
        valueLeft[subject] = dict()
        valueRight[subject] = dict()
        trials = distLeft[subject].keys()
        for trial in trials:
            valueLeft[subject][trial] = np.absolute((np.absolute(
                distLeft[subject][trial])-15)/5)
            valueRight[subject][trial] = np.absolute((np.absolute(
                distRight[subject][trial])-15)/5)

    print("Starting grid search...")
    rangeD = [0.0045, 0.005, 0.0055]
    rangeTheta = [0.25, 0.3, 0.35]
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
        print("Running subject " + subject + "...")
        trials = rt[subject].keys()
        trialSet = np.random.choice(trials, trialsPerSubject, replace=False)
        for trial in trialSet:
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
    distSecondFix = evenDists.distSecondFix
    distMiddleFix = dists.distMiddleFix

    generate_probabilistic_simulations(probLeftFixFirst, distTransition,
        distFirstFix, distSecondFix, distMiddleFix, posteriors)


if __name__ == '__main__':
    main()
