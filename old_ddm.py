#!/usr/bin/python

# old_ddm.py
# Author: Gabriela Tavares, gtavares@caltech.edu

# Old implementation of the traditional drift-diffusion model (DDM). This
# algorithm uses reaction time histograms conditioned on choice from both data
# and simulations to estimate each model's likelihood. Here we perforrm a test
# to check the validity of this algorithm. Artificil data is generated using
# specific parameters for the model. These parameters are then recovered through
# a maximum likelihood estimation procedure, using a grid search over the 2 free
# parameters of the model.

from multiprocessing import Pool

import collections
import numpy as np


def ddm(d, sigma, valueLeft, valueRight, timeStep=10, barrier=1):
    # DDM algorithm. Given the parameters of the model and the trial conditions,
    # returns the choice and reaction time as predicted by the model.
    # Args:
    #   d: float, parameter of the model which controls the speed of integration
    #       of the signal.
    #   sigma: float, parameter of the model, controls the Gaussian noise to be
    #       added to the RDV signal.
    #   valueLeft: integer, value of the left item.
    #   valueRight: integer, value of the right item.
    #   timeStep: integer, value in miliseconds which determines how often the
    #       RDV signal is updated.
    #   barrier: positive number, magnitude of the signal thresholds.

    rt = 0
    choice = 0
    rdv = 0
    valueDiff = valueLeft - valueRight

    while rdv < barrier and rdv > -barrier:
        rt = rt + timeStep
        epsilon = np.random.normal(0, sigma)
        rdv = rdv + (d * valueDiff) + epsilon

    if rdv >= barrier:
        choice = -1
    elif rdv <= -barrier:
        choice = 1

    results = collections.namedtuple('Results', ['rt', 'choice'])
    return results(rt, choice)


def get_model_likelihood(d, sigma, trialConditions, numSimulations, histBins,
    dataHistLeft, dataHistRight):
    # Computes the likelihood of a data set given the parameters of the DDM.
    # Data set is provided in the form of reaction time histograms conditioned
    # on choice.
    # Args:
    #   d: float, parameter of the model which controls the speed of integration
    #       of the signal.
    #   sigma: float, parameter of the model, standard deviation for the normal
    #       distribution.
    #   trialConditions: list of pairs corresponding to the different trial
    #       conditions. Each pair contains the values of left and right items.
    #   numSimulations: integer, number of simulations per trial condition to be
    #       generated when creating reaction time histograms.
    #   histBins: list of numbers corresponding to the time bins used to create
    #       the reaction time histograms.
    #   dataHistLeft: dict indexed by trial condition (where each trial
    #       condition is a pair (valueLeft, valueRight)). Each entry is a numpy
    #       array corresponding to the reaction time histogram conditioned on
    #       left choice for the data. It is assumed that this histogram was
    #       created using the same time bins as argument histBins.
    #   dataHistRight: same as dataHistLeft, except that the reaction time
    #       histograms are conditioned on right choice.
    #   Returns:
    #       The likelihood for the given data and model.

    likelihood = 0
    for trialCondition in trialConditions:
        rtsLeft = list()
        rtsRight = list()
        sim = 0
        while sim < numSimulations:
            results = ddm(d, sigma, trialCondition[0], trialCondition[1])
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
    # Wrapper for get_model_likelihood() which takes a single argument. Intended
    # for parallel computation using a thread pool.
    # Args:
    #   params: tuple consisting of all arguments required by
    #       get_model_likelihood().
    # Returns:
    #   The output of get_model_likelihood().

    return get_model_likelihood(*params)


def main():
    numThreads = 9
    pool = Pool(numThreads)

    # Parameters for artificial data.
    d = 0.006
    sigma = 0.08

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
            results = ddm(d, sigma, trialCondition[0], trialCondition[1])
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
    rangeSigma = [0.06, 0.08, 0.1]
    models = list()
    for d in rangeD:
        for sigma in rangeSigma:
            model = (d, sigma)
            models.append(model)

    listParams = list()
    for model in models:
        listParams.append((model[0], model[1], trialConditions, numSimulations,
            histBins, dataHistLeft, dataHistRight))
    likelihoods = pool.map(get_model_likelihood_wrapper, listParams)

    for i in xrange(len(models)):
        print("L" + str(models[i]) + " = " + str(likelihoods[i]))
    bestIndex = likelihoods.index(max(likelihoods))
    print("Best fit: " + str(models[bestIndex]))


if __name__ == '__main__':
    main()
