#!/usr/bin/python

# simulate_addm_true_distributions.py
# Author: Gabriela Tavares, gtavares@caltech.edu

from multiprocessing import Pool

import collections
import numpy as np
import pandas as pd
import sys

from addm import get_empirical_distributions
from util import load_data_from_csv, save_simulations_to_csv


def run_simulations(probLeftFixFirst, distLatencies, distTransitions,
    distFixations, numTrials, trialConditions, bins, numFixDists, d, theta, 
    std=0, mu=0, timeStep=10, barrier=1, visualDelay=0, motorDelay=0):
    if std == 0:
        if mu != 0:
            std = mu * d
        else:
            return None

    # Simulation data to be returned.
    rt = dict()
    choice = dict()
    valueLeft = dict()
    valueRight = dict()
    fixItem = dict()
    fixTime = dict()
    fixRDV = dict()
    uninterruptedLastFixTime = dict()

    trialCount = 0

    for trialCondition in trialConditions:
        vLeft = trialCondition[0]
        vRight = trialCondition[1]
        fixUnfixValueDiffs = {1: vLeft - vRight, 2: vRight - vLeft}
        trial = 0
        while trial < numTrials:
            fixItem[trialCount] = list()
            fixTime[trialCount] = list()
            fixRDV[trialCount] = list()

            RDV = 0
            trialTime = 0

            # Sample and iterate over the latency for this trial.
            trialAborted = False
            while True:
                latency = np.random.choice(distLatencies)
                for t in xrange(int(latency // timeStep)):
                    # Sample the change in RDV from the distribution.
                    RDV += np.random.normal(0, std)
                    # If the RDV hit one of the barriers, we abort the trial,
                    # since a trial must end on an item fixation.
                    if RDV >= barrier or RDV <= -barrier:
                        trialAborted = True
                        break

                if trialAborted:
                    RDV = 0
                    trialAborted = False
                    continue
                else:
                    # Add latency to this trial's data.
                    fixRDV[trialCount].append(RDV)
                    fixItem[trialCount].append(0)
                    fixTime[trialCount].append(latency - (latency % timeStep))
                    trialTime += latency - (latency % timeStep)
                    break

            # Sample the first fixation for this trial.
            probLeftRight = np.array([probLeftFixFirst, 1 - probLeftFixFirst])
            currFixItem = np.random.choice([1, 2], p=probLeftRight)
            valueDiff = fixUnfixValueDiffs[currFixItem]
            prob = ([value for (key, value) in
                sorted(distFixations[1][valueDiff].items())])
            currFixTime = np.random.choice(bins, p=prob) - visualDelay

            # Iterate over all fixations in this trial.
            fixNumber = 2
            trialFinished = False
            trialAborted = False
            while True:
                # Iterate over the visual delay for the current fixation.
                for t in xrange(int(visualDelay // timeStep)):
                    # Sample the change in RDV from the distribution.
                    RDV += np.random.normal(0, std)

                    # If the RDV hit one of the barriers, the trial is over.
                    if RDV >= barrier or RDV <= -barrier:
                        if RDV >= barrier:
                            choice[trialCount] = -1
                        elif RDV <= -barrier:
                            choice[trialCount] = 1
                        valueLeft[trialCount] = vLeft
                        valueRight[trialCount] = vRight
                        fixRDV[trialCount].append(RDV)
                        fixItem[trialCount].append(currFixItem)
                        fixTime[trialCount].append(((t + 1) * timeStep) +
                            motorDelay)
                        trialTime += ((t + 1) * timeStep) + motorDelay
                        rt[trialCount] = trialTime
                        uninterruptedLastFixTime[trialCount] = currFixTime
                        trialFinished = True
                        break

                if trialFinished:
                    break

                # Iterate over the time interval of the current fixation.
                for t in xrange(int(currFixTime // timeStep)):
                    # We use a distribution to model changes in RDV
                    # stochastically. The mean of the distribution (the change
                    # most likely to occur) is calculated from the model
                    # parameters and from the values of the two items.
                    if currFixItem == 1:  # Subject is looking left.
                        mean = d * (vLeft - (theta * vRight))
                    elif currFixItem == 2:  # Subject is looking right.
                        mean = d * (-vRight + (theta * vLeft))

                    # Sample the change in RDV from the distribution.
                    RDV += np.random.normal(mean, std)

                    # If the RDV hit one of the barriers, the trial is over.
                    if RDV >= barrier or RDV <= -barrier:
                        if RDV >= barrier:
                            choice[trialCount] = -1
                        elif RDV <= -barrier:
                            choice[trialCount] = 1
                        valueLeft[trialCount] = vLeft
                        valueRight[trialCount] = vRight
                        fixRDV[trialCount].append(RDV)
                        fixItem[trialCount].append(currFixItem)
                        fixTime[trialCount].append(((t + 1) * timeStep) +
                            visualDelay + motorDelay)
                        trialTime += (((t + 1) * timeStep) + visualDelay +
                            motorDelay)
                        rt[trialCount] = trialTime
                        uninterruptedLastFixTime[trialCount] = currFixTime
                        trialFinished = True
                        break

                if trialFinished:
                    break

                # Add previous fixation to this trial's data.
                fixRDV[trialCount].append(RDV)
                fixItem[trialCount].append(currFixItem)
                fixTime[trialCount].append((currFixTime -
                    (currFixTime % timeStep)) + visualDelay)
                trialTime += ((currFixTime - (currFixTime % timeStep)) + 
                    visualDelay)

                # Sample and iterate over transition.
                transition = np.random.choice(distTransitions)
                for t in xrange(int(transition // timeStep)):
                    # Sample the change in RDV from the distribution.
                    RDV += np.random.normal(0, std)

                    # If the RDV hit one of the barriers, we abort the trial,
                    # since a trial must end on an item fixation.
                    if RDV >= barrier or RDV <= -barrier:
                        trialFinished = True
                        trialAborted = True
                        break

                if trialFinished:
                    break

                # Add previous transition to this trial's data.
                fixRDV[trialCount].append(RDV)
                fixItem[trialCount].append(0)
                fixTime[trialCount].append(transition - (transition % timeStep))
                trialTime += transition - (transition % timeStep)

                # Sample the next fixation for this trial.
                if currFixItem == 1:
                    currFixItem = 2
                elif currFixItem == 2:
                    currFixItem = 1
                valueDiff = fixUnfixValueDiffs[currFixItem]
                prob = ([value for (key, value) in
                    sorted(distFixations[fixNumber][valueDiff].items())])
                currFixTime = np.random.choice(bins, p=prob) - visualDelay
                if fixNumber < numFixDists:
                    fixNumber += 1

            # Move on to the next trial.
            if not trialAborted:
                trial += 1
                trialCount += 1

    simul = collections.namedtuple('Simul', ['rt', 'choice', 'valueLeft',
        'valueRight', 'fixItem', 'fixTime', 'fixRDV',
        'uninterruptedLastFixTime'])
    return simul(rt, choice, valueLeft, valueRight, fixItem, fixTime, fixRDV,
        uninterruptedLastFixTime)


def main():
    # Time bins to be used in the fixation distributions.
    binStep = 10
    bins = range(binStep, 3000 + binStep, binStep)

    numFixDists = 3  # Number of fixation distributions.
    N = 2  # Number of iterations to approximate true distributions.

    # Load experimental data from CSV file.
    data = load_data_from_csv("expdata.csv", "fixations.csv", True)
    rt = data.rt
    choice = data.choice
    valueLeft = data.valueLeft
    valueRight = data.valueRight
    fixItem = data.fixItem
    fixTime = data.fixTime

    # Get empirical distributions.
    dists = get_empirical_distributions(valueLeft, valueRight, fixItem, fixTime,
        useOddTrials=False, useEvenTrials=True)
    probLeftFixFirst = dists.probLeftFixFirst
    distLatencies = dists.distLatencies
    distTransitions = dists.distTransitions
    distFixations = dists.distFixations

    # Parameters for generating simulations.
    d = 0.004
    std = 0.07
    theta = 0.25
    numTrials = 400
    orientations = range(-15,20,5)
    trialConditions = list()
    for oLeft in orientations:
        for oRight in orientations:
            if oLeft != oRight:
                vLeft = np.absolute((np.absolute(oLeft) - 15) / 5)
                vRight = np.absolute((np.absolute(oRight) - 15) / 5)
                trialConditions.append((vLeft, vRight))

    # Create original empirical distributions of fixations.
    empiricalFixDist = dict()
    for numFix in xrange(1, numFixDists + 1):
        empiricalFixDist[numFix] = dict()
        for valueDiff in xrange(-3,4):
            empiricalFixDist[numFix][valueDiff] = dict()
            for bin in bins:
                empiricalFixDist[numFix][valueDiff][bin] = 0
            for fixTime in distFixations[numFix][valueDiff]:
                bin = min((fixTime // binStep) + 1, len(bins)) * binStep
                empiricalFixDist[numFix][valueDiff][bin] += 1

    # Normalize the distributions.
    for numFix in xrange(1, numFixDists + 1):
        for valueDiff in xrange(-3,4):
            sumBins = sum(empiricalFixDist[numFix][valueDiff].values())
            for bin in bins:
                empiricalFixDist[numFix][valueDiff][bin] = (
                    float(empiricalFixDist[numFix][valueDiff][bin]) /
                    float(sumBins))

    # Repeat the process N times.
    for i in xrange(N):
        # Generate simulations using the current empirical distributions and the
        # model parameters.
        simul = run_simulations(probLeftFixFirst, distLatencies,
            distTransitions, empiricalFixDist, numTrials, trialConditions, bins,
            numFixDists, d, theta, std=std)
        simulRt = simul.rt
        simulChoice = simul.choice
        simulValueLeft = simul.valueLeft
        simulValueRight = simul.valueRight
        simulFixItem = simul.fixItem
        simulFixTime = simul.fixTime
        simulFixRDV = simul.fixRDV
        simulUninterruptedLastFixTime = simul.uninterruptedLastFixTime

        countLastFix = dict()
        countTotal = dict()
        for numFix in xrange(1, numFixDists + 1):
            countLastFix[numFix] = dict()
            countTotal[numFix] = dict()
            for valueDiff in xrange(-3,4):
                countLastFix[numFix][valueDiff] = dict()
                countTotal[numFix][valueDiff] = dict()
                for bin in bins:
                    countLastFix[numFix][valueDiff][bin] = 0
                    countTotal[numFix][valueDiff][bin] = 0

        for trial in simulRt.keys():
            # Count all item fixations, except last.
            fixUnfixValueDiffs = {1: simulValueLeft[trial] - 
                simulValueRight[trial], 2: simulValueRight[trial] -
                simulValueLeft[trial]}
            numFix = 1
            for item, time in zip(simulFixItem[trial][:-1],
                simulFixTime[trial][:-1]):
                if item == 1 or item == 2:
                    bin = min((time // binStep) + 1, len(bins)) * binStep
                    vDiff = fixUnfixValueDiffs[item]
                    countTotal[numFix][vDiff][bin] += 1
                    if numFix < numFixDists:
                        numFix += 1
            # Count last fixation.
            item = simulFixItem[trial][-1]
            vDiff = fixUnfixValueDiffs[item]
            bin = min((simulUninterruptedLastFixTime[trial] // binStep) + 1,
                len(bins)) * binStep
            countLastFix[numFix][vDiff][bin] += 1
            countTotal[numFix][vDiff][bin] += 1

        # Obtain true distributions of fixations.
        trueFixDist = dict()
        for numFix in xrange(1, numFixDists + 1):
            trueFixDist[numFix] = dict()
            for valueDiff in xrange(-3,4):
                trueFixDist[numFix][valueDiff] = dict()
                for bin in bins:
                    probNotLastFix = 1
                    if countTotal[numFix][valueDiff][bin] > 0:
                        probNotLastFix = 1 - (
                            float(countLastFix[numFix][valueDiff][bin])
                            / float(countTotal[numFix][valueDiff][bin]))
                    if probNotLastFix == 0:
                        trueFixDist[numFix][valueDiff][bin] = (
                            empiricalFixDist[numFix][valueDiff][bin])
                    else:
                        trueFixDist[numFix][valueDiff][bin] = (
                            float(empiricalFixDist[numFix][valueDiff][bin]) /
                            float(probNotLastFix))
        # Normalize the distributions.
        for numFix in xrange(1, numFixDists + 1):
            for valueDiff in xrange(-3,4):
                sumBins = sum(trueFixDist[numFix][valueDiff].values())
                if sumBins > 0:
                    for bin in bins:
                        trueFixDist[numFix][valueDiff][bin] = (
                            float(trueFixDist[numFix][valueDiff][bin]) /
                            float(sumBins))

        # Update empirical distributions using the current true distributions.
        empiricalFixDist = trueFixDist

    # Generate final simulations.
    simul = run_simulations(probLeftFixFirst, distLatencies, distTransitions,
        empiricalFixDist, numTrials, trialConditions, bins, numFixDists, d,
        theta, std=std)
    simulRt = simul.rt
    simulChoice = simul.choice
    simulValueLeft = simul.valueLeft
    simulValueRight = simul.valueRight
    simulFixItem = simul.fixItem
    simulFixTime = simul.fixTime
    simulFixRDV = simul.fixRDV
    simulUninterruptedLastFixTime = simul.uninterruptedLastFixTime

    totalTrials = numTrials * len(trialConditions)
    save_simulations_to_csv(simulChoice, simulRt, simulValueLeft,
        simulValueRight, simulFixItem, simulFixTime, simulFixRDV, totalTrials)


if __name__ == '__main__':
    main()