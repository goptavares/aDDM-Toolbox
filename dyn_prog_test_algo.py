#!/usr/bin/python

# dyn_prog_test_algo.py
# Author: Gabriela Tavares, gtavares@caltech.edu

from multiprocessing import Pool

import csv
import numpy as np
import operator

from dyn_prog_fixations import (load_data_from_csv, analysis_per_trial,
    get_empirical_distributions, run_simulations)


def run_analysis(numTrials, rt, choice, valueLeft, valueRight, fixItem, fixTime,
    d, theta, mu, verbose=True):
    logLikelihood = 0
    for trial in xrange(numTrials):
        if verbose and trial % 100 == 0:
            print("Trial " + str(trial) + "/" + str(numTrials) + "...")
        logLikelihood += np.log(analysis_per_trial(rt[trial], choice[trial],
            valueLeft[trial], valueRight[trial], fixItem[trial], fixTime[trial],
            d, theta, mu=mu, plotResults=False))
    return -logLikelihood


def run_analysis_wrapper(params):
    return run_analysis(*params)


def main():
    numThreads = 8
    pool = Pool(numThreads)

    # Load experimental data from CSV file.
    data = load_data_from_csv("expdata.csv", "fixations.csv")
    rt = data.rt
    choice = data.choice
    valueLeft = data.valueLeft
    valueRight = data.valueRight
    fixItem = data.fixItem
    fixTime = data.fixTime

    # Get empirical distributions.
    dists = get_empirical_distributions(rt, choice, valueLeft, valueRight,
        fixItem, fixTime)
    probLeftFixFirst = dists.probLeftFixFirst
    distTransition = dists.distTransition
    distFirstFix = dists.distFirstFix
    distMiddleFix = dists.distMiddleFix

    # Parameters for artificial data generation.
    numTrials = 360
    d = 0.001
    theta = 0.3
    mu = 50

    orientations = range(-15,20,5)
    trialConditions = list()
    for oLeft in orientations:
        for oRight in orientations:
            if oLeft != oRight:
                trialConditions.append((oLeft, oRight))

    # Generate artificial data.
    print("Running simulations...")
    simul = run_simulations(probLeftFixFirst, distTransition, distFirstFix,
        distMiddleFix, numTrials, trialConditions, d, theta, mu=mu)
    simulRt = simul.rt
    simulChoice = simul.choice
    simulDistLeft = simul.distLeft
    simulDistRight = simul.distRight
    simulFixItem = simul.fixItem
    simulFixTime = simul.fixTime

    # Get item values.
    totalTrials = numTrials * len(trialConditions)
    simulValueLeft = dict()
    simulValueRight = dict()
    for trial in xrange(totalTrials):
        simulValueLeft[trial] = np.absolute((np.absolute(
            simulDistLeft[trial])-15)/5)
        simulValueRight[trial] = np.absolute((np.absolute(
            simulDistRight[trial])-15)/5)

    # Write artificial data to CSV.
    with open("expdata_" + str(d) + "_" + str(theta) + "_" + str(mu) + "_" +
        str(numTrials) + ".csv", "wb") as csvFile:
        csvWriter = csv.writer(csvFile, delimiter=',', quotechar='|',
            quoting=csv.QUOTE_MINIMAL)
        csvWriter.writerow(["parcode", "trial", "rt", "choice", "dist_left",
            "dist_right"])
        for trial in xrange(totalTrials):
            csvWriter.writerow(["dummy_subj", str(trial), str(simulRt[trial]),
                str(simulChoice[trial]), str(simulDistLeft[trial]),
                str(simulDistRight[trial])])

    with open("fixations_" + str(d) + "_" + str(theta) + "_" + str(mu) + "_" +
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
    rangeD = [0.0002, 0.0004, 0.0006]
    rangeTheta = [0.3, 0.5, 0.7]
    rangeMu = [20, 50, 80]

    models = list()
    list_params = list()
    results = list()
    for d in rangeD:
        for theta in rangeTheta:
            for mu in rangeMu:
                models.append((d, theta, mu))
                params = (totalTrials, simulRt, simulChoice, simulValueLeft,
                    simulValueRight, simulFixItem, simulFixTime, d, theta, mu)
                list_params.append(params)

    print("Starting pool of workers...")
    results = pool.map(run_analysis_wrapper, list_params)

    # Get optimal parameters.
    minNegLogLikeIdx = results.index(min(results))
    optimD = models[minNegLogLikeIdx][0]
    optimTheta = models[minNegLogLikeIdx][1]
    optimMu = models[minNegLogLikeIdx][2]

    print("Finished grid search!")
    print("Optimal d: " + str(optimD))
    print("Optimal theta: " + str(optimTheta))
    print("Optimal mu: " + str(optimMu))
    print("Min NLL: " + str(min(results)))
 

if __name__ == '__main__':
    main()
