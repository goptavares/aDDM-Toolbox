#!/usr/bin/python

# dyn_prog_test_algo.py
# Author: Gabriela Tavares, gtavares@caltech.edu

import collections
import numpy as np
import pandas as pd

from dyn_prog_fixations import load_data_from_csv


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


def generate_fake_data(probLeftFixFirst, distTransition, distFirstFix,
    distMiddleFix):
    timeStep = 10
    numTrials = 1
    L = 1
    d = 0.00015
    theta = 0.7
    mu = 100
    sigma = mu * d

    orientations = range(-15,20,5)
    trialConditions = list()
    for oLeft in orientations:
        for oRight in orientations:
            trialConditions.append((oLeft, oRight))

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

                    if currFixItem == -1:
                        delta = vLeft - (theta * vRight)
                    elif currFixItem == 1:
                        delta = - (vRight - (theta * vLeft))

                    # Sample random white Gaussian noise.
                    epsilon = np.random.normal(0, sigma)

                    # Update RDV.
                    RDV += d * delta + epsilon

                    trialTime += timeStep

                if trialFinished:
                    break

                # Add last fixation to this trial's data.
                fixItem[trialCount].append(currFixItem)
                fixTime[trialCount].append(t * timeStep)

                # Sample next fixation for this trial.
                currFixItem = -1 * currFixItem
                currFixTime = np.random.choice(distMiddleFix[valueDiff])

    simul = collections.namedtuple('Simul', ['rt', 'choice', 'valueLeft',
        'valueRight', 'fixItem', 'fixTime'])
    return simul(rt, choice, valueLeft, valueRight, fixItem, fixTime)


def main():
    data = load_data_from_csv()
    rt = data.rt
    choice = data.choice
    valueLeft = data.valueLeft
    valueRight = data.valueRight
    fixItem = data.fixItem
    fixTime = data.fixTime

    dists = get_empirical_distributions(rt, choice, valueLeft, valueRight,
        fixItem, fixTime)
    probLeftFixFirst = dists.probLeftFixFirst
    distTransition = dists.distTransition
    distFirstFix = dists.distFirstFix
    distMiddleFix = dists.distMiddleFix

    simul = generate_fake_data(probLeftFixFirst, distTransition, distFirstFix,
        distMiddleFix)
    simulRt = simul.rt
    simulChoice = simul.choice
    simulValueLeft = simul.valueLeft
    simulValueRight = simul.valueRight
    simulFixItem = simul.fixItem
    simulFixTime = simul.fixTime


if __name__ == '__main__':
    main()