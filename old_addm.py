#!/usr/bin/python

"""
old_addm.py
Author: Gabriela Tavares, gtavares@caltech.edu

Old implementation of the attentional drift-diffusion model (aDDM). This
algorithm uses reaction time histograms conditioned on choice from both data and
simulations to estimate each model's likelihood. Here we perforrm a test to
check the validity of this algorithm. Artificil data is generated using specific
parameters for the model. These parameters are then recovered through a maximum
likelihood estimation procedure, using a grid search over the 3 free parameters
of the model.
"""

from multiprocessing import Pool

import argparse
import collections
import numpy as np

from addm import get_empirical_distributions
from util import load_data_from_csv


def addm(probLeftFixFirst, distLatencies, distTransitions, distFixations, d,
         sigma, theta, valueLeft, valueRight, timeStep=10, barrier=1,
         numFixDists=3):
    """
    DDM algorithm. Given the parameters of the model and the trial conditions,
    returns the choice and reaction time as predicted by the model.
    Args:
      probLeftFixFirst: float between 0 and 1, empirical probability that the
          left item will be fixated first.
      distLatencies: numpy array corresponding to the empirical distribution of
          trial latencies (delay before first fixation) in miliseconds.
      distTransitions: numpy array corresponding to the empirical distribution
          of transitions (delays between item fixations) in miliseconds.
      distFixations: dict whose indexing is controlled by argument fixDistType.
          Its entries are numpy arrays corresponding to the empirical
          distributions of item fixation durations in miliseconds.
      d: float, parameter of the model which controls the speed of integration
          of the signal.
      sigma: float, parameter of the model, controls the Gaussian noise to be
          added to the RDV signal.
      theta: float between 0 and 1, parameter of the model which controls the
          attentional bias.
      valueLeft: integer, value of the left item.
      valueRight: integer, value of the right item.
      timeStep: integer, value in miliseconds which determines how often the RDV
          signal is updated.
      barrier: positive number, magnitude of the signal thresholds.
      numFixDists: integer, number of fixation types to use in the fixation
          distributions. For instance, if numFixDists equals 3, then 3 separate
          fixation types will be used, corresponding to the 1st, 2nd and other
          (3rd and up) fixations in each trial.
    Returns:
      A named tuple containing the following fields:
        RT: integer, reaction time in miliseconds.
        choice: either -1 (for left item) or +1 (for right item).
    """

    RDV = 0
    latency = (np.random.choice(distLatencies) // timeStep) * timeStep
    for t in xrange(int(latency // timeStep)):
        RDV += np.random.normal(0, sigma)

    RT = latency
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
            if RDV >= barrier or RDV <= -barrier:
                if RDV >= barrier:
                    choice = -1
                elif RDV <= -barrier:
                    choice = 1
                decisionReached = True
                break

            RT = RT + timeStep
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
            RT += transition
            if currFixItem == 1:
                currFixItem = 2
            elif currFixItem == 2:
                currFixItem = 1
            valueDiff = fixUnfixValueDiffs[currFixItem]
            currFixTime = np.random.choice(distFixations[fixNumber][valueDiff])
            if fixNumber < numFixDists:
                fixNumber += 1

    results = collections.namedtuple('Results', ['RT', 'choice'])
    return results(RT, choice)


def get_model_likelihood(d, sigma, theta, trialConditions,
                         numSimulations, histBins, dataHistLeft, dataHistRight,
                         probLeftFixFirst, distLatencies, distTransitions,
                         distFixations):
    """
    Computes the likelihood of a data set given the parameters of the aDDM. Data
    set is provided in the form of reaction time histograms conditioned on
    choice.
    Args:
      d: float, parameter of the model which controls the speed of integration
          of the signal.
      sigma: float, parameter of the model, standard deviation for the normal
          distribution.
      theta: float between 0 and 1, parameter of the model which controls the
          attentional bias.
      trialConditions: list of pairs corresponding to the different trial
          conditions. Each pair contains the values of left and right items.
      numSimulations: integer, number of simulations per trial condition to be
          generated when creating reaction time histograms.
      histBins: list of numbers corresponding to the time bins used to create
          the reaction time histograms.
      dataHistLeft: dict indexed by trial condition (where each trial condition
          is a pair (valueLeft, valueRight)). Each entry is a numpy array
          corresponding to the reaction time histogram conditioned on left
          choice for the data. It is assumed that this histogram was created
          using the same time bins as argument histBins.
      dataHistRight: same as dataHistLeft, except that the reaction time
          histograms are conditioned on right choice.
      probLeftFixFirst: float between 0 and 1, empirical probability that the
          left item will be fixated first.
      distLatencies: numpy array corresponding to the empirical distribution of
          trial latencies (delay before first fixation) in miliseconds.
      distTransitions: numpy array corresponding to the empirical distribution
          of transitions (delays between item fixations) in miliseconds.
      distFixations: dict whose indexing is controlled by argument fixDistType.
          Its entries are numpy arrays corresponding to the empirical
          distributions of item fixation durations in miliseconds.
      Returns:
          The likelihood for the given data and model.
    """

    likelihood = 0
    for trialCondition in trialConditions:
        RTsLeft = list()
        RTsRight = list()
        sim = 0
        while sim < numSimulations:
            results = addm(probLeftFixFirst, distLatencies, distTransitions,
                           distFixations, d, sigma, theta, trialCondition[0],
                           trialCondition[1])
            if results.choice == -1:
                RTsLeft.append(results.RT)
            elif results.choice == 1:
                RTsRight.append(results.RT)
            sim += 1

        simulLeft = np.histogram(RTsLeft, bins=histBins)[0]
        if np.sum(simulLeft) != 0:
            simulLeft = simulLeft / float(np.sum(simulLeft))
        logSimulLeft = np.where(simulLeft > 0, np.log(simulLeft), 0)
        dataLeft = np.array(dataHistLeft[trialCondition])
        likelihood += np.dot(logSimulLeft, dataLeft)

        simulRight = np.histogram(RTsRight, bins=histBins)[0]
        if np.sum(simulRight) != 0:
            simulRight = simulRight / float(np.sum(simulRight))
        logSimulRight = np.where(simulRight > 0, np.log(simulRight), 0)
        dataRight = np.array(dataHistRight[trialCondition])
        likelihood += np.dot(logSimulRight, dataRight)

    return likelihood


def get_model_likelihood_wrapper(params):
    """
    Wrapper for get_model_likelihood() which takes a single argument. Intended
    for parallel computation using a thread pool.
    Args:
      params: tuple consisting of all arguments required by
          get_model_likelihood().
    Returns:
      The output of get_model_likelihood().
    """

    return get_model_likelihood(*params)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-threads", type=int, default=9,
                        help="size of the thread pool")
    parser.add_argument("--num-trials", type=int, default=10,
                        help="number of artificial data trials to be generated "
                        "per trial condition")
    parser.add_argument("--num-simulations", type=int, default=10,
                        help="number of simulations to be generated per trial "
                        "condition, to be used in the RT histograms")
    parser.add_argument("--verbose", default=False, action="store_true",
                        help="increase output verbosity")
    args = parser.parse_args()

    pool = Pool(args.num_threads)

    # Load experimental data from CSV file.
    data = load_data_from_csv("expdata.csv", "fixations.csv",
                              useAngularDists=True)

    # Get empirical distributions.
    dists = get_empirical_distributions(
        data.valueLeft, data.valueRight, data.fixItem, data.fixTime)
    probLeftFixFirst = dists.probLeftFixFirst
    distLatencies = dists.distLatencies
    distTransitions = dists.distTransitions
    distFixations = dists.distFixations

    if args.verbose:
        print("Done getting empirical distributions!")

    # Parameters for artificial data.
    d = 0.006
    sigma = 0.08
    theta = 0.5

    maxRT = 8000
    histBins = range(0, maxRT + 100, 100)
    histBins = histBins + [20000]

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
        RTsLeft = list()
        RTsRight = list()
        trial = 0
        while trial < args.num_trials:
            results = addm(probLeftFixFirst, distLatencies, distTransitions,
                           distFixations, d, sigma, theta, trialCondition[0],
                           trialCondition[1])
            if results.choice == -1:
                RTsLeft.append(results.RT)
            elif results.choice == 1:
                RTsRight.append(results.RT)
            trial += 1
        dataHistLeft[trialCondition] = np.histogram(RTsLeft, bins=histBins)[0]
        dataHistRight[trialCondition] = np.histogram(RTsRight, bins=histBins)[0]

    if args.verbose:
        print("Done generating histograms of artificial data!")
    
    # Grid search on the parameters of the model.
    rangeD = [0.002, 0.006, 0.01]
    rangeSigma = [0.04, 0.08, 0.12]
    rangeTheta = [0.1, 0.5, 0.9]
    models = list()
    for d in rangeD:
        for sigma in rangeSigma:
            for theta in rangeTheta:
                model = (d, sigma, theta)
                models.append(model)

    listParams = list()
    for model in models:
        listParams.append(
            (model[0], model[1], model[2], trialConditions,
            args.num_simulations, histBins, dataHistLeft, dataHistRight,
            probLeftFixFirst, distLatencies, distTransitions, distFixations))
    likelihoods = pool.map(get_model_likelihood_wrapper, listParams)

    if args.verbose:
        for i in xrange(len(models)):
            print("L" + str(models[i]) + " = " + str(likelihoods[i]))
        bestIndex = likelihoods.index(max(likelihoods))
        print("Best fit: " + str(models[bestIndex]))


if __name__ == '__main__':
    main()
