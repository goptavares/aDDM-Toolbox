#!/usr/bin/python

# individual_fitting.py
# Author: Gabriela Tavares, gtavares@caltech.edu

import matplotlib
matplotlib.use('Agg')

from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import operator
import pandas as pd

from handle_fixations import (load_data_from_csv, analysis_per_trial,
    get_empirical_distributions, run_simulations)
from group_fitting import (generate_choice_curves, generate_rt_curves,
    save_simulations_to_csv)


def run_analysis(rt, choice, valueLeft, valueRight, fixItem, fixTime, d, theta,
    std, useOddTrials=True, useEvenTrials=True, verbose=True):
    trialsPerSubject = 200
    logLikelihood = 0
    subjects = rt.keys()
    for subject in subjects:
        trials = rt[subject].keys()
        trialSet = np.random.choice(trials, trialsPerSubject, replace=False)
        for trial in trialSet:
            if not useOddTrials and trial % 2 != 0:
                continue
            if not useEvenTrials and trial % 2 == 0:
                continue
            likelihood = analysis_per_trial(rt[subject][trial],
                choice[subject][trial], valueLeft[subject][trial],
                valueRight[subject][trial], fixItem[subject][trial],
                fixTime[subject][trial], d, theta, std=std)
            if likelihood != 0:
                logLikelihood += np.log(likelihood)

    if verbose:
        print("NLL for " + str(d) + ", " + str(theta) + ", "
            + str(std) + ": " + str(-logLikelihood))
    return -logLikelihood


def run_analysis_wrapper(params):
    return run_analysis(*params)


def main():
    numThreads = 9
    pool = Pool(numThreads)

    subject = "pai"
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

    # Maximum likelihood estimation using odd trials only.
    # Grid search on the parameters of the model.
    print("Starting grid search for subject " + subject + "...")
    rangeD = [0.005, 0.006, 0.007]
    rangeTheta = [0.1, 0.2, 0.3]
    rangeStd = [0.08, 0.09, 0.1]

    models = list()
    listParams = list()
    for d in rangeD:
        for theta in rangeTheta:
            for std in rangeStd:
                models.append((d, theta, std))
                params = (rt, choice, valueLeft, valueRight, fixItem, fixTime,
                    d, theta, std, True, False)
                listParams.append(params)

    results = pool.map(run_analysis_wrapper, listParams)

    # Get optimal parameters.
    minNegLogLikeIdx = results.index(min(results))
    optimD = models[minNegLogLikeIdx][0]
    optimTheta = models[minNegLogLikeIdx][1]
    optimStd = models[minNegLogLikeIdx][2]
    print("Finished coarse grid search!")
    print("Optimal d: " + str(optimD))
    print("Optimal theta: " + str(optimTheta))
    print("Optimal std: " + str(optimStd))
    print("Min NLL: " + str(min(results)))

    # Get empirical distributions from even trials.
    evenDists = get_empirical_distributions(rt, choice, distLeft, distRight,
        fixItem, fixTime, useOddTrials=False, useEvenTrials=True)
    probLeftFixFirst = evenDists.probLeftFixFirst
    distTransition = evenDists.distTransition
    distFirstFix = evenDists.distFirstFix
    distSecondFix = evenDists.distSecondFix
    distThirdFix = evenDists.distThirdFix
    distOtherFix = evenDists.distOtherFix

    # Parameters for generating simulations.
    numTrials = 32
    orientations = range(-15,20,5)
    trialConditions = list()
    for oLeft in orientations:
        for oRight in orientations:
            if oLeft != oRight:
                trialConditions.append((oLeft, oRight))

    # Generate simulations using the even trials distributions and the
    # estimated parameters.
    simul = run_simulations(probLeftFixFirst, distTransition, distFirstFix,
        distSecondFix, distThirdFix, distOtherFix, numTrials, trialConditions,
        optimD, optimTheta, std=optimStd)
    simulRt = simul.rt
    simulChoice = simul.choice
    simulDistLeft = simul.distLeft
    simulDistRight = simul.distRight
    simulFixItem = simul.fixItem
    simulFixTime = simul.fixTime
    simulFixRDV = simul.fixRDV

    # Get item values for simulations.
    totalTrials = numTrials * len(trialConditions)
    simulValueLeft = dict()
    simulValueRight = dict()
    for trial in xrange(totalTrials):
        simulValueLeft[trial] = np.absolute((np.absolute(
            simulDistLeft[trial])-15)/5)
        simulValueRight[trial] = np.absolute((np.absolute(
            simulDistRight[trial])-15)/5)

    # Create pdf file to save figures.
    pp = PdfPages("figures_" + str(optimD) + "_" + str(optimTheta) + "_" +
        str(optimStd) + "_" + str(numTrials) + ".pdf")

    # Generate choice and rt curves for real data (odd trials) and
    # simulations (generated from even trials).
    fig1 = generate_choice_curves(choice, valueLeft, valueRight, simulChoice,
        simulValueLeft, simulValueRight, totalTrials)
    pp.savefig(fig1)
    fig2 = generate_rt_curves(rt, valueLeft, valueRight, simulRt,
        simulValueLeft, simulValueRight, totalTrials)
    pp.savefig(fig2)
    pp.close()

    save_simulations_to_csv(simulChoice, simulRt, simulValueLeft,
        simulValueRight, simulFixItem, simulFixTime, simulFixRDV, totalTrials)


if __name__ == '__main__':
    main()