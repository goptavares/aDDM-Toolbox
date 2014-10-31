#!/usr/bin/python

# dyn_prog_test_algo.py
# Author: Gabriela Tavares, gtavares@caltech.edu

from multiprocessing import Pool

import collections
import numpy as np
import operator

from dyn_prog_fixations import load_data_from_csv, analysis_per_trial


def get_empirical_distributions(rt, choice, valueLeft, valueRight, fixItem,
    fixTime, useOddTrials=True, useEvenTrials=True):
    valueDiffs = range(-6,8,2)

    countLeftFirst = 0
    countTotalTrials = 0
    distTransitionList = list()
    distFirstFixList = list()
    distMiddleFixList = dict()
    for valueDiff in valueDiffs:
        distMiddleFixList[valueDiff] = list()

    subjects = rt.keys()
    for subject in subjects:
        trials = rt[subject].keys()
        for trial in trials:
            if not useOddTrials and trial % 2 != 0:
                continue
            if not useEvenTrials and trial % 2 == 0:
                continue

            # Get value difference between best and worst items for this trial.
            valueDiff = (max(valueLeft[subject][trial],
                valueRight[subject][trial]) - min(valueLeft[subject][trial],
                valueRight[subject][trial]))
            # Iterate over this trial's fixations.
            firstFix = True
            transitionTime = 0
            for i in xrange(fixItem[subject][trial].shape[0]):
                item = fixItem[subject][trial][i]
                if item != 1 and item != 2:
                    transitionTime += fixTime[subject][trial][i]
                else:
                    if firstFix:
                        firstFix = False
                        distFirstFixList.append(fixTime[subject][trial][i])
                        countTotalTrials +=1
                        if item == 1:  # First fixation was left.
                            countLeftFirst +=1
                    else:
                        distMiddleFixList[valueDiff].append(
                            fixTime[subject][trial][i])
            # Add transition time for this trial to distribution.
            distTransitionList.append(transitionTime)

    probLeftFixFirst = float(countLeftFirst) / float(countTotalTrials)
    distTransition = np.array(distTransitionList)
    distFirstFix = np.array(distFirstFixList)
    distMiddleFix = dict()
    for valueDiff in valueDiffs:
        distMiddleFix[valueDiff] = np.array(distMiddleFixList[valueDiff])

    dists = collections.namedtuple('Dists', ['probLeftFixFirst',
        'distTransition', 'distFirstFix', 'distMiddleFix'])
    return dists(probLeftFixFirst, distTransition, distFirstFix, distMiddleFix)


def run_simulations(numTrials, trialConditions, d, theta, mu, probLeftFixFirst,
    distTransition, distFirstFix, distMiddleFix):
    timeStep = 10
    L = 1

    # Simulation data to be returned.
    rt = dict()
    choice = dict()
    valueLeft = dict()
    valueRight = dict()
    fixItem = dict()
    fixTime = dict()

    trialCount = 0

    for trialCondition in trialConditions:
        vLeft = (-np.absolute(trialCondition[0]) / 2.5) + 3
        vRight = (-np.absolute(trialCondition[1]) / 2.5) + 3
        valueDiff = max(vLeft, vRight) - min(vLeft, vRight)
        for trial in xrange(numTrials):
            fixItem[trialCount] = list()
            fixTime[trialCount] = list()

            # Sample transition time from the empirical distribution.
            transitionTime = np.random.choice(distTransition)
            fixItem[trialCount].append(0)
            fixTime[trialCount].append(transitionTime)

            # Sample the first fixation for this trial.
            probLeftRight = np.array([probLeftFixFirst, 1-probLeftFixFirst])
            currFixItem = np.random.choice([1, 2], p=probLeftRight)
            currFixTime = np.random.choice(distFirstFix)

            # Iterate over all fixations in this trial.
            RDV = 0
            trialTime = 0
            trialFinished = False
            while True:
                # Iterate over the time interval of the current fixation.
                for t in xrange(1, int(currFixTime // timeStep) + 1):
                    # We use a distribution to model changes in RDV
                    # stochastically. The mean of the distribution (the change
                    # most likely to occur) is calculated from the model
                    # parameters and from the values of the two items.
                    if currFixItem == 1:  # Subject is looking left.
                        mean = d * (vLeft - (theta * vRight))
                    elif currFixItem == 2:  # Subject is looking right.
                        mean = d * (-vRight + (theta * vLeft))

                    # Sample the change in RDV from the distribution.
                    std = mu * d
                    RDV += np.random.normal(mean, std)

                    trialTime += timeStep

                    # If the RDV hit one of the barriers, the trial is over.
                    if RDV > L or RDV < -L:
                        if RDV > L:
                            choice[trialCount] = -1
                        elif RDV < -L:
                            choice[trialCount] = 1
                        rt[trialCount] = transitionTime + trialTime
                        valueLeft[trialCount] = vLeft
                        valueRight[trialCount] = vRight
                        fixItem[trialCount].append(currFixItem)
                        fixTime[trialCount].append(t * timeStep)
                        trialCount += 1
                        trialFinished = True
                        break

                if trialFinished:
                    break

                # Add previous fixation to this trial's data.
                fixItem[trialCount].append(currFixItem)
                fixTime[trialCount].append(currFixTime)

                # Sample next fixation for this trial.
                if currFixItem == 1:
                    currFixItem = 2
                elif currFixItem == 2:
                    currFixItem = 1
                currFixTime = np.random.choice(distMiddleFix[valueDiff])

    simul = collections.namedtuple('Simul', ['rt', 'choice', 'valueLeft',
        'valueRight', 'fixItem', 'fixTime'])
    return simul(rt, choice, valueLeft, valueRight, fixItem, fixTime)


def run_analysis(numTrials, rt, choice, valueLeft, valueRight, fixItem, fixTime,
    d, theta, mu, verbose=True):
    logLikelihood = 0
    for trial in xrange(numTrials):
        if verbose and trial % 100 == 0:
            print("Trial " + str(trial) + "/" + str(numTrials) + "...")
        logLikelihood += np.log(analysis_per_trial(rt[trial], choice[trial],
            valueLeft[trial], valueRight[trial], fixItem[trial], fixTime[trial],
            d, theta, mu, plotResults=False))
    return logLikelihood


def run_analysis_wrapper(params):
    return run_analysis(*params)


def main():
    numThreads = 8
    pool = Pool(numThreads)

    # Load experimental data from CSV file.
    data = load_data_from_csv()
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

    # Parameters for fake data generation.
    numTrials = 100
    d = 0.0003
    theta = 0.3
    mu = 400

    orientations = range(-15,20,5)
    trialConditions = list()
    for oLeft in orientations:
        for oRight in orientations:
            if oLeft != oRight:
                trialConditions.append((oLeft, oRight))

    # Generate fake data.
    print("Running simulations...")
    simul = run_simulations(numTrials, trialConditions, d, theta, mu,
        probLeftFixFirst, distTransition, distFirstFix, distMiddleFix)
    simulRt = simul.rt
    simulChoice = simul.choice
    simulValueLeft = simul.valueLeft
    simulValueRight = simul.valueRight
    simulFixItem = simul.fixItem
    simulFixTime = simul.fixTime

    # Grid search to recover the parameters.
    rangeD = [0.00025, 0.0003, 0.00035]
    rangeTheta = [0.2, 0.3, 0.4]
    rangeMu = [350, 400, 450]

    totalTrials = numTrials * len(trialConditions)
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
    max_likelihood_idx = results.index(max(results))
    optimD = models[max_likelihood_idx][0]
    optimTheta = models[max_likelihood_idx][1]
    optimMu = models[max_likelihood_idx][2]

    print("Finished grid search!")
    print("Optimal d: " + str(optimD))
    print("Optimal theta: " + str(optimTheta))
    print("Optimal mu: " + str(optimMu))
 

if __name__ == '__main__':
    main()
