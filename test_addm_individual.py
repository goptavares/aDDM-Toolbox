#!/usr/bin/python

# test_addm_individual.py
# Author: Gabriela Tavares, gtavares@caltech.edu

from multiprocessing import Pool

import csv
import numpy as np

from addm import (analysis_per_trial, get_empirical_distributions,
    run_simulations)
from util import load_data_from_csv


def run_analysis(params):
    return analysis_per_trial(*params)


def main():
    numThreads = 9
    pool = Pool(numThreads)

    subject = 'lua'
    valueLeft = dict()
    valueRight = dict()
    fixItem = dict()
    fixTime = dict()

    # Load experimental data from CSV file.
    data = load_data_from_csv("expdata.csv", "fixations.csv", True)
    valueLeft[subject] = data.valueLeft[subject]
    valueRight[subject] = data.valueRight[subject]
    fixItem[subject] = data.fixItem[subject]
    fixTime[subject] = data.fixTime[subject]

    # Get empirical distributions.
    dists = get_empirical_distributions(valueLeft, valueRight, fixItem, fixTime)
    probLeftFixFirst = dists.probLeftFixFirst
    distLatencies = dists.distLatencies
    distTransitions = dists.distTransitions
    distFixations = dists.distFixations

    # Parameters for artificial data generation.
    numTrials = 200
    d = 0.006
    theta = 0.5
    std = 0.07

    orientations = range(-15,20,5)
    trialConditions = list()
    for oLeft in orientations:
        for oRight in orientations:
            if oLeft != oRight:
                vLeft = np.absolute((np.absolute(oLeft) - 15) / 5)
                vRight = np.absolute((np.absolute(oRight) - 15) / 5)
                trialConditions.append((vLeft, vRight))

    # Generate artificial data.
    print("Running simulations...")
    simul = run_simulations(probLeftFixFirst, distLatencies, distTransitions,
        distFixations, numTrials, trialConditions, d, theta, std=std)
    simulRt = simul.rt
    simulChoice = simul.choice
    simulValueLeft = simul.valueLeft
    simulValueRight = simul.valueRight
    simulFixItem = simul.fixItem
    simulFixTime = simul.fixTime

    # Write artificial data to CSV.
    with open("expdata_" + str(d) + "_" + str(theta) + "_" + str(std) + "_" +
        str(numTrials) + ".csv", "wb") as csvFile:
        csvWriter = csv.writer(csvFile, delimiter=',', quotechar='|',
            quoting=csv.QUOTE_MINIMAL)
        csvWriter.writerow(["parcode", "trial", "rt", "choice", "value_left",
            "value_right"])
        for trial in xrange(totalTrials):
            csvWriter.writerow(["dummy_subj", str(trial), str(simulRt[trial]),
                str(simulChoice[trial]), str(simulValueLeft[trial]),
                str(simulValueRight[trial])])

    with open("fixations_" + str(d) + "_" + str(theta) + "_" + str(std) + "_" +
        str(numTrials) + ".csv", "wb") as csvFile:
        csvWriter = csv.writer(csvFile, delimiter=',', quotechar='|',
            quoting=csv.QUOTE_MINIMAL)
        csvWriter.writerow(["parcode", "trial", "fix_item", "fix_time"])
        for trial in xrange(totalTrials):
            for fix in xrange(len(simulFixItem[trial])):
                csvWriter.writerow(["dummy_subj", str(trial),
                    str(simulFixItem[trial][fix]),
                    str(simulFixTime[trial][fix])])

    # Grid search to recover the parameters.
    print("Starting grid search...")
    rangeD = [0.0055, 0.006, 0.0065]
    rangeTheta = [0.3, 0.5, 0.7]
    rangeStd = [0.065, 0.07, 0.075]
    numModels = len(rangeD) * len(rangeTheta) * len(rangeStd)

    models = list()
    posteriors = dict()
    for d in rangeD:
        for theta in rangeTheta:
            for std in rangeStd:
                model = (d, theta, std)
                models.append(model)
                posteriors[model] = 1./ numModels

    trials = simulChoice.keys()
    for trial in trials:
        listParams = list()
        for model in models:
            listParams.append((simulChoice[trial], simulValueLeft[trial],
                simulValueRight[trial], simulFixItem[trial],
                simulFixTime[trial], model[0], model[1], model[2]))
        likelihoods = pool.map(run_analysis, listParams)

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

        if trial % 200 == 0:
            for model in posteriors:
                print("P" + str(model) + " = " + str(posteriors[model]))
            print("Sum: " + str(sum(posteriors.values())))
 
    for model in posteriors:
        print("P" + str(model) + " = " + str(posteriors[model]))
    print("Sum: " + str(sum(posteriors.values())))


if __name__ == '__main__':
    main()
