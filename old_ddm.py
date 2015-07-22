#!/usr/bin/python

# old_ddm.py
# Author: Gabriela Tavares, gtavares@caltech.edu

from multiprocessing import Pool

import collections
import numpy as np


def ddm(L, d, sigma, valueLeft, valueRight, timeStep):
    rt = 0
    choice = 0
    RDV = 0
    valueDiff = valueLeft - valueRight

    while RDV < L and RDV > -L:
        rt = rt + timeStep
        epsilon = np.random.normal(0, sigma)
        RDV = RDV + (d * valueDiff) + epsilon

    if RDV >= L:
        choice = -1
    elif RDV <= -L:
        choice = 1

    results = collections.namedtuple('Results', ['rt', 'choice'])
    return results(rt, choice)


def get_model_likelihood(L, d, sigma, timeStep, trialConditions, numSimulations,
    histBins, dataHistLeft, dataHistRight):
    likelihood = 0
    for trialCondition in trialConditions:
        rtsLeft = list()
        rtsRight = list()
        sim = 0
        while sim < numSimulations:
            results = ddm(L, d, sigma, trialCondition[0], trialCondition[1],
                timeStep)
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

    L = 1
    d = 0.006
    sigma = 0.08
    timeStep = 10

    maxRt = 8000
    histBins = range(0, maxRt + 100, 100)
    histBins = histBins + [20000]

    numTrials = 210
    numSimulations = 210

    numValues = 4
    values = range(1, numValues + 1, 1)
    trialConditions = list()
    for vLeft in values:
        for vRight in values:
            trialConditions.append((vLeft, vRight))

    # Generate artificial data.
    dataRtLeft = dict()
    dataRtRight = dict()
    for trialCondition in trialConditions:
        dataRtLeft[trialCondition] = list()
        dataRtRight[trialCondition] = list()
    for trialCondition in trialConditions:
        trial = 0
        while trial < numTrials:
            results = ddm(L, d, sigma, trialCondition[0], trialCondition[1],
                timeStep)
            rt = results.rt
            choice = results.choice
            if choice == -1:
                dataRtLeft[trialCondition].append(rt)
            elif choice == 1:
                dataRtRight[trialCondition].append(rt)
            trial += 1

    # Generate histograms for artificial data.
    dataHistLeft = dict()
    dataHistRight = dict()
    for trialCondition in trialConditions:
        dataHistLeft[trialCondition] = np.histogram(dataRtLeft[trialCondition],
            bins=histBins)[0]
        dataHistRight[trialCondition] = np.histogram(
            dataRtRight[trialCondition], bins=histBins)[0]

    # Grid search on the parameters of the model.
    rangeD = [0.004, 0.006, 0.008]
    rangeStd = [0.06, 0.08, 0.1]
    models = list()
    for d in rangeD:
        for std in rangeStd:
            model = (d, std)
            models.append(model)

    listParams = list()
    for model in models:
        listParams.append((L, model[0], model[1], timeStep, trialConditions,
            numSimulations, histBins, dataHistLeft, dataHistRight))
    likelihoods = pool.map(get_model_likelihood_wrapper, listParams)

    for i in xrange(len(models)):
        print("L" + str(models[i]) + " = " + str(likelihoods[i]))
    bestIndex = likelihoods.index(max(likelihoods))
    print("Best fit: " + str(models[bestIndex]))


if __name__ == '__main__':
    main()
