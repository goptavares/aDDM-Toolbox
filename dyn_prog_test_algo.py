#!/usr/bin/python

# dyn_prog_test_algo.py
# Author: Gabriela Tavares, gtavares@caltech.edu

from multiprocessing import Pool

import collections
import numpy as np
import operator
import pandas as pd

from dyn_prog_fixations import load_data_from_csv, analysis_per_trial


def get_empirical_distributions(rt, choice, valueLeft, valueRight, fixItem,
    fixTime):
    valueDiffs = xrange(-3,4,1)

    countLeftFirst = 0
    countTotalTrials = 0
    distTransitionList = list()
    distFirstFixList = dict()
    distMiddleFixList = dict()
    for valueDiff in valueDiffs:
        distFirstFixList[valueDiff] = list()
        distMiddleFixList[valueDiff] = list()

    subjects = rt.keys()
    for subject in subjects:
        trials = rt[subject].keys()
        for trial in trials:
            # Get value difference for this trial.
            valueDiff = valueLeft[subject][trial] - valueRight[subject][trial]
            # Iterate over this trial's fixations.
            firstFix = True
            transitionTime = 0
            # for item in np.nditer(fixItem[subject][trial]):
            for i in xrange(fixItem[subject][trial].shape[0]):
                item = fixItem[subject][trial][i]
                if item != 1 and item != 2:
                    transitionTime += fixTime[subject][trial][i]
                else:
                    if firstFix:
                        firstFix = False
                        distFirstFixList[valueDiff].append(
                            fixTime[subject][trial][i])
                        countTotalTrials +=1
                        if item == 1:  # first fixation was left
                            countLeftFirst +=1
                    else:
                        distMiddleFixList[valueDiff].append(
                            fixTime[subject][trial][i])
            # Add transition time for this trial to distribution.
            distTransitionList.append(transitionTime)

    probLeftFixFirst = countLeftFirst / countTotalTrials
    distTransition = np.array(distTransitionList)
    distFirstFix = dict()
    distMiddleFix = dict()
    for valueDiff in valueDiffs:
        distFirstFix[valueDiff] = np.array(distFirstFixList[valueDiff])
        distMiddleFix[valueDiff] = np.array(distMiddleFixList[valueDiff])

    dists = collections.namedtuple('Dists', ['probLeftFixFirst',
        'distTransition', 'distFirstFix', 'distMiddleFix'])
    return dists(probLeftFixFirst, distTransition, distFirstFix, distMiddleFix)


def generate_fake_data(numTrials, trialConditions, d, theta, std,
    probLeftFixFirst, distTransition, distFirstFix, distMiddleFix):
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
        for trial in xrange(numTrials):
            vLeft = np.absolute((np.absolute(trialCondition[0])-15)/5)
            vRight = np.absolute((np.absolute(trialCondition[1])-15)/5)
            valueDiff = vLeft - vRight
            RDV = 0
            fixItem[trialCount] = list()
            fixTime[trialCount] = list()

            # Sample transition time from the empirical distribution.
            transitionTime = np.random.choice(distTransition)
            fixItem[trialCount].append(0)
            fixTime[trialCount].append(transitionTime)

            # Sample the first fixation for this trial.
            probLeftRight = np.array([probLeftFixFirst, 1-probLeftFixFirst])
            currFixItem = np.random.choice([-1, 1], p=probLeftRight)
            currFixTime = np.random.choice(distFirstFix[valueDiff])

            # Iterate over all fixations in this trial.
            trialTime = 0
            trialFinished = False
            while True:
                # Iterate over the time interval of the current fixation.
                for t in xrange(1, int(currFixTime/timeStep + 1)):
                    if RDV > L or RDV < -L:
                        if RDV > L:
                            choice[trialCount] = -1
                        elif RDV < -L:
                            choice[trialCount] = 1
                        rt[trialCount]  = trialTime
                        valueLeft[trialCount] = vLeft
                        valueRight[trialCount] = vRight
                        fixItem[trialCount].append(currFixItem)
                        fixTime[trialCount].append((t-1) * timeStep)
                        trialCount += 1
                        trialFinished = True
                        break

                    # We use a distribution to model changes in RDV
                    # stochastically. The mean of the distribution (the change
                    # most likely to occur) is calculated from the model
                    # parameters and from the values of the two items.
                    if currFixItem == -1:  # subject is looking left.
                        mean = d * (vLeft - (theta * vRight))
                    elif currFixItem == 1:  # subject is looking right.
                        mean = d * (-vRight + (theta * vLeft))

                    # Sample the change in RDV from the distribution.
                    RDV += np.random.normal(mean, std)

                    trialTime += timeStep

                if trialFinished:
                    break

                # Add previous fixation to this trial's data.
                fixItem[trialCount].append(currFixItem)
                fixTime[trialCount].append(t * timeStep)

                # Sample next fixation for this trial.
                currFixItem = -1 * currFixItem
                currFixTime = np.random.choice(distMiddleFix[valueDiff])

    simul = collections.namedtuple('Simul', ['rt', 'choice', 'valueLeft',
        'valueRight', 'fixItem', 'fixTime'])
    return simul(rt, choice, valueLeft, valueRight, fixItem, fixTime)


def run_analysis(numTrials, rt, choice, valueLeft, valueRight, fixItem, fixTime,
    d, theta, std):
    likelihood = 0
    for trial in xrange(numTrials):
        likelihood += analysis_per_trial(rt[trial], choice[trial],
            valueLeft[trial], valueRight[trial], fixItem[trial], fixTime[trial],
            d, theta, std)
    print("Likelihood: " + str(likelihood))
    return likelihood


def run_analysis_wrapper(params):
    return run_analysis(*params)


def main():
    numThreads = 4
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
    d = 0.03
    theta = 0.7
    std = 0.05

    orientations = range(-15,20,5)
    trialConditions = list()
    for oLeft in orientations:
        for oRight in orientations:
            if oLeft != oRight:
                trialConditions.append((oLeft, oRight))

    # Generate fake data.
    simul = generate_fake_data(numTrials, trialConditions, d, theta, std,
        probLeftFixFirst, distTransition, distFirstFix, distMiddleFix)
    simulRt = simul.rt
    simulChoice = simul.choice
    simulValueLeft = simul.valueLeft
    simulValueRight = simul.valueRight
    simulFixItem = simul.fixItem
    simulFixTime = simul.fixTime

    # Grid search to recover the parameters.
    rangeD = [0.03, 0.04, 0.05]
    rangeTheta = [0.5, 0.7, 0.9]
    rangeStd = [0.04, 0.05, 0.06]

    totalTrials = numTrials * len(trialConditions)
    models = list()
    list_params = list()
    for d in rangeD:
        for theta in rangeTheta:
            for std in rangeStd:
                models.append((d, theta, std))
                params = (totalTrials, simulRt, simulChoice, simulValueLeft,
                    simulValueRight, simulFixItem, simulFixTime, d, theta, std)
                list_params.append(params)

    print("Starting pool of workers...")
    results = pool.map(run_analysis_wrapper, list_params)

    # Get optimal parameters.
    max_likelihood_idx = results.index(max(results))
    optimD = models[max_likelihood_idx][0]
    optimTheta = models[max_likelihood_idx][1]
    optimStd = models[max_likelihood_idx][2]

    print("Finished grid search!")
    print("Optimal d: " + str(optimD))
    print("Optimal theta: " + str(optimTheta))
    print("Optimal std: " + str(optimStd))
 

if __name__ == '__main__':
    main()
