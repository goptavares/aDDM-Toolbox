#!/usr/bin/python

# old_addm.py
# Author: Gabriela Tavares, gtavares@caltech.edu

from multiprocessing import Pool

import collections
import numpy as np

from addm import get_empirical_distributions
from util import load_data_from_csv


def addm(L, d, sigma, theta, valueLeft, valueRight, timeStep, probLeftFixFirst,
    distLatencies, distTransitions, distFixations, numFixDists=3):
    RDV = 0
    latency = (np.random.choice(distLatencies) // timeStep) * timeStep
    for t in xrange(int(latency // timeStep)):
        RDV += np.random.normal(0, sigma)

    rt = latency
    choice = 0
    fixUnfixValueDiffs = {1: valueLeft - valueRight, 2: valueRight - valueLeft}
    
    probLeftRight = np.array([probLeftFixFirst, 1-probLeftFixFirst])
    currFixItem = np.random.choice([1, 2], p=probLeftRight)
    valueDiff = fixUnfixValueDiffs[currFixItem]
    currFixTime = np.random.choice(distFixations[1][valueDiff])

    decisionReached = False
    fixNumber = 2
    while True:
        for t in xrange(int(currFixTime // timeStep)):
            if RDV >= L or RDV <= -L:
                if RDV >= L:
                    choice = -1
                elif RDV <= -L:
                    choice = 1
                decisionReached = True
                break

            rt = rt + timeStep
            epsilon = np.random.normal(0, sigma)
            if currFixItem == 1:
                RDV = RDV + (d * (valueLeft - (theta * valueRight))) + epsilon
            elif currFixItem == 2:
                RDV = RDV + (d * (-valueRight + (theta * valueLeft))) + epsilon

        if decisionReached:
            break
        else:
            transition = ((np.random.choice(distTransitions) // timeStep) *
                timeStep)
            for t in xrange(int(transition // timeStep)):
                RDV += np.random.normal(0, sigma)
            rt += transition
            if currFixItem == 1:
                currFixItem = 2
            elif currFixItem == 2:
                currFixItem = 1
            valueDiff = fixUnfixValueDiffs[currFixItem]
            currFixTime = np.random.choice(distFixations[fixNumber][valueDiff])
            if fixNumber < numFixDists:
                fixNumber += 1

    results = collections.namedtuple('Results', ['rt', 'choice'])
    return results(rt, choice)


def get_model_likelihood(L, d, sigma, theta, timeStep, trialConditions,
    numSimulations, histBins, dataHistLeft, dataHistRight, probLeftFixFirst,
    distLatencies, distTransitions, distFixations):
    likelihood = 0
    for trialCondition in trialConditions:
        rtsLeft = list()
        rtsRight = list()
        sim = 0
        while sim < numSimulations:
            results = addm(L, d, sigma, theta, trialCondition[0],
                trialCondition[1], timeStep, probLeftFixFirst, distLatencies,
                distTransitions, distFixations)
            if results.choice == -1:
                rtsLeft.append(results.rt)
            elif results.choice == 1:
                rtsRight.append(results.rt)
            sim += 1

        simulLeft = np.histogram(rtsLeft, bins=histBins)[0]
        if np.sum(simulLeft) != 0:
            simulLeft = simulLeft / float(np.sum(simulLeft))
        logSimulLeft = np.where(simulLeft > 0, np.log(simulLeft), 0)
        dataLeft = np.array(dataHistLeft[trialCondition])
        likelihood += np.dot(logSimulLeft, dataLeft)

        simulRight = np.histogram(rtsRight, bins=histBins)[0]
        if np.sum(simulRight) != 0:
            simulRight = simulRight / float(np.sum(simulRight))
        logSimulRight = np.where(simulRight > 0, np.log(simulRight), 0)
        dataRight = np.array(dataHistRight[trialCondition])
        likelihood += np.dot(logSimulRight, dataRight)

    return likelihood


def get_model_likelihood_wrapper(params):
    return get_model_likelihood(*params)


def main():
    numThreads = 9
    pool = Pool(numThreads)

    # Load experimental data from CSV file.
    data = load_data_from_csv("expdata.csv", "fixations.csv", True)

    # Get empirical distributions.
    dists = get_empirical_distributions(data.valueLeft, data.valueRight,
        data.fixItem, data.fixTime)
    probLeftFixFirst = dists.probLeftFixFirst
    distLatencies = dists.distLatencies
    distTransitions = dists.distTransitions
    distFixations = dists.distFixations

    print("Done getting empirical distributions!")

    L = 1
    d = 0.006
    sigma = 0.08
    theta = 0.5
    timeStep = 10

    maxRt = 8000
    histBins = range(0, maxRt + 100, 100)
    histBins = histBins + [20000]

    numTrials = 1000
    numSimulations = 10000

    orientations = range(-15,20,5)
    trialConditions = list()
    for oLeft in orientations:
        for oRight in orientations:
            if oLeft != oRight:
                vLeft = np.absolute((np.absolute(oLeft) - 15) / 5)
                vRight = np.absolute((np.absolute(oRight) - 15) / 5)
                trialConditions.append((vLeft, vRight))

    # Generate histograms for artificial data.
    dataHistLeft = dict()
    dataHistRight = dict()
    for trialCondition in trialConditions:
        rtsLeft = list()
        rtsRight = list()
        trial = 0
        while trial < numTrials:
            results = addm(L, d, sigma, theta, trialCondition[0],
                trialCondition[1], timeStep, probLeftFixFirst, distLatencies,
                distTransitions, distFixations)
            if results.choice == -1:
                rtsLeft.append(results.rt)
            elif results.choice == 1:
                rtsRight.append(results.rt)
            trial += 1
        dataHistLeft[trialCondition] = np.histogram(rtsLeft, bins=histBins)[0]
        dataHistRight[trialCondition] = np.histogram(rtsRight, bins=histBins)[0]

    print("Done generating histograms of artificial data!")
    
    # Grid search on the parameters of the model.
    rangeD = [0.002, 0.006, 0.01]
    rangeStd = [0.04, 0.08, 0.12]
    rangeTheta = [0.1, 0.5, 0.9]
    models = list()
    for d in rangeD:
        for std in rangeStd:
            for theta in rangeTheta:
                model = (d, std, theta)
                models.append(model)

    listParams = list()
    for model in models:
        listParams.append((L, model[0], model[1], model[2], timeStep,
            trialConditions, numSimulations, histBins, dataHistLeft,
            dataHistRight, probLeftFixFirst, distLatencies, distTransitions,
            distFixations))
    likelihoods = pool.map(get_model_likelihood_wrapper, listParams)

    for i in xrange(len(models)):
        print("L" + str(models[i]) + " = " + str(likelihoods[i]))
    bestIndex = likelihoods.index(max(likelihoods))
    print("Best fit: " + str(models[bestIndex]))


if __name__ == '__main__':
    main()
