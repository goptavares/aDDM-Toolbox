#!/usr/bin/python

# cis_trans_fitting.py
# Author: Gabriela Tavares, gtavares@caltech.edu

# Maximum likelihood estimation procedure for the attentional drift-diffusion
# model (aDDM), specific for perceptual decisions, allowing for analysis of cis
# trials or trans trials exclusively. Function main() takes two boolean
# arguments which determine whether cis (1st argument) or trans (2nd argument)
# trials should be used. A grid search is performed over the 3 free parameters
# of the model. Data from all subjects is pooled such that a single set of
# optimal parameters is estimated. aDDM simulations are generated for the model
# estimated.

from multiprocessing import Pool

import numpy as np
import sys

from addm import (analysis_per_trial, get_empirical_distributions,
    run_simulations)
from util import load_data_from_csv, save_simulations_to_csv


def run_analysis(choice, valueLeft, valueRight, fixItem, fixTime, d, theta, std,
    useOddTrials=True, useEvenTrials=True, isCisTrial=None, isTransTrial=None,
    useCisTrials=True, useTransTrials=True, verbose=True):
    # Computes the negative log likelihood of a data set given the parameters of
    # the aDDM.
    # Args:
    #   choice: dict of dicts with same indexing as rt. Each entry is an integer
    #       corresponding to the decision made in that trial.
    #   valueLeft: dict of dicts with same indexing as rt. Each entry is an
    #       integer corresponding to the value of the left item.
    #   valueRight: dict of dicts with same indexing as rt. Each entry is an
    #       integer corresponding to the value of the right item.
    #   fixItem: dict of dicts with same indexing as rt. Each entry is an
    #       ordered list of fixated items in the trial.
    #   fixTime: dict of dicts with same indexing as rt. Each entry is an
    #       ordered list of fixation durations in the trial.
    #   d: float, parameter of the model which controls the speed of integration
    #       of the signal.
    #   theta: float between 0 and 1, parameter of the model which controls the
    #       attentional bias.
    #   std: float, parameter of the model, standard deviation for the normal
    #       distribution.
    #   useOddTrials: boolean, whether or not to use odd trials in the analysis.
    #   useEvenTrials: boolean, whether or not to use even trials in the
    #       analysis.
    #   isCisTrial: dict of dicts, indexed first by subject then by trial
    #       number. Each entry is a boolean indicating if the trial is cis (both
    #       bars on the same side of the target).
    #   isTransTrial: dict of dicts with same indexing as isCisTrial. Each entry
    #       is a boolean indicating if the trial is trans (bars on either side
    #       of the target).
    #   useCisTrials: boolean, whether or not to use cis trials  in the
    #       analysis.
    #   useTransTrials: boolean, whether or not to use trans trials in the
    #       analysis.
    #   verbose: boolean, whether or not to print updates during computation.
    # Returns:
    #   The negative log likelihood for the given data set and model.

    logLikelihood = 0
    subjects = choice.keys()
    for subject in subjects:
        if verbose:
            print("Running subject " + subject + "...")
        trials = choice[subject].keys()
        for trial in trials:
            if not useOddTrials and trial % 2 != 0:
                continue
            if not useEvenTrials and trial % 2 == 0:
                continue
            if (not useCisTrials and isCisTrial[subject][trial] and
                not isTransTrial[subject][trial]):
                continue
            if (not useTransTrials and isTransTrial[subject][trial] and
                not isCisTrial[subject][trial]):
                continue
            likelihood = analysis_per_trial(choice[subject][trial],
                valueLeft[subject][trial], valueRight[subject][trial],
                fixItem[subject][trial], fixTime[subject][trial], d, theta,
                std=std)
            if likelihood != 0:
                logLikelihood += np.log(likelihood)

    if verbose:
        print("NLL for " + str(d) + ", " + str(theta) + ", "
            + str(std) + ": " + str(-logLikelihood))
    return -logLikelihood


def run_analysis_wrapper(params):
    # Wrapper for run_analysis() which takes a single argument. Intended for
    # parallel computation using a thread pool.
    # Args:
    #   params: tuple consisting of all arguments required by run_analysis().
    # Returns:
    #   The output of run_analysis().

    return run_analysis(*params)


def main(argv):
    useCisTrials = argv[0]
    useTransTrials = argv[1]

    numThreads = 9
    pool = Pool(numThreads)

    # Load experimental data from CSV file.
    data = load_data_from_csv("expdata.csv", "fixations.csv", True)
    choice = data.choice
    valueLeft = data.valueLeft
    valueRight = data.valueRight
    fixItem = data.fixItem
    fixTime = data.fixTime
    isCisTrial = data.isCisTrial
    isTransTrial = data.isTransTrial

    # Maximum likelihood estimation.
    # Grid search on the parameters of the model using odd trials only.
    print("Starting grid search...")
    rangeD = [0.004, 0.005, 0.006]
    rangeTheta = [0.3, 0.5, 0.7]
    rangeStd = [0.04, 0.065, 0.09]

    models = list()
    listParams = list()
    for d in rangeD:
        for theta in rangeTheta:
            for std in rangeStd:
                models.append((d, theta, std))
                params = (choice, valueLeft, valueRight, fixItem, fixTime, d,
                    theta, std, True, False, isCisTrial, isTransTrial,
                    useCisTrials, useTransTrials)
                listParams.append(params)

    print("Starting pool of workers...")
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

    # Get empirical distributions from even trials only.
    evenDists = get_empirical_distributions(valueLeft, valueRight, fixItem,
        fixTime, useOddTrials=False, useEvenTrials=True, isCisTrial=isCisTrial,
        isTransTrial=isTransTrial, useCisTrials=useCisTrials,
        useTransTrials=useTransTrials)
    probLeftFixFirst = evenDists.probLeftFixFirst
    distLatencies = evenDists.distLatencies
    distTransitions = evenDists.distTransitions
    distFixations = evenDists.distFixations

    # Parameters for generating simulations.
    numTrials = 400
    orientations = range(-15,20,5)
    trialConditions = list()
    for oLeft in orientations:
        for oRight in orientations:
            vLeft = np.absolute((np.absolute(oLeft) - 15) / 5)
            vRight = np.absolute((np.absolute(oRight) - 15) / 5)
            if oLeft != oRight and useCisTrials and oLeft * oRight >= 0:
                trialConditions.append((vLeft, vRight))
            elif oLeft != oRight and useTransTrials and oLeft * oRight <= 0:
                trialConditions.append((vLeft, vRight))

    # Generate simulations using the empirical distributions and the
    # estimated parameters.
    simul = run_simulations(probLeftFixFirst, distLatencies, distTransitions,
        distFixations, numTrials, trialConditions, optimD, optimTheta,
        std=optimStd)
    simulRt = simul.rt
    simulChoice = simul.choice
    simulValueLeft = simul.valueLeft
    simulValueRight = simul.valueRight
    simulFixItem = simul.fixItem
    simulFixTime = simul.fixTime
    simulFixRDV = simul.fixRDV

    totalTrials = numTrials * len(trialConditions)
    save_simulations_to_csv(simulChoice, simulRt, simulValueLeft,
        simulValueRight, simulFixItem, simulFixTime, simulFixRDV, totalTrials)


if __name__ == '__main__':
    main(sys.argv[1:])
