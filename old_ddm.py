#!/usr/bin/python

"""
old_ddm.py
Author: Gabriela Tavares, gtavares@caltech.edu

Old implementation of the traditional drift-diffusion model (DDM). This
algorithm uses reaction time histograms conditioned on choice from both data and
simulations to estimate each model's likelihood. Here we perforrm a test to
check the validity of this algorithm. Artificil data is generated using specific
parameters for the model. These parameters are then recovered through a maximum
likelihood estimation procedure, using a grid search over the 2 free parameters
of the model.
"""

from multiprocessing import Pool

import argparse
import collections
import numpy as np


def ddm(d, sigma, valueLeft, valueRight, timeStep=10, barrier=1):
    """
    DDM algorithm. Given the parameters of the model and the trial conditions,
    returns the choice and reaction time as predicted by the model.
    Args:
      d: float, parameter of the model which controls the speed of integration
          of the signal.
      sigma: float, parameter of the model, controls the Gaussian noise to be
          added to the RDV signal.
      valueLeft: integer, value of the left item.
      valueRight: integer, value of the right item.
      timeStep: integer, value in miliseconds which determines how often the RDV
          signal is updated.
      barrier: positive number, magnitude of the signal thresholds.
    Returns:
      A named tuple containing the following fields:
        RT: integer, reaction time in miliseconds.
        choice: either -1 (for left item) or +1 (for right item).
    """

    RT = 0
    choice = 0
    RDV = 0
    valueDiff = valueLeft - valueRight

    while RDV < barrier and RDV > -barrier:
        RT = RT + timeStep
        epsilon = np.random.normal(0, sigma)
        RDV = RDV + (d * valueDiff) + epsilon

    if RDV >= barrier:
        choice = -1
    elif RDV <= -barrier:
        choice = 1

    results = collections.namedtuple('Results', ['RT', 'choice'])
    return results(RT, choice)


def get_model_likelihood(d, sigma, trialConditions, numSimulations, histBins,
                         dataHistLeft, dataHistRight):
    """
    Computes the likelihood of a data set given the parameters of the DDM. Data
    set is provided in the form of reaction time histograms conditioned on
    choice.
    Args:
      d: float, parameter of the model which controls the speed of integration
          of the signal.
      sigma: float, parameter of the model, standard deviation for the normal
          distribution.
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
      Returns:
          The likelihood for the given data and model.
    """

    likelihood = 0
    for trialCondition in trialConditions:
        RTsLeft = list()
        RTsRight = list()
        sim = 0
        while sim < numSimulations:
            try:
                results = ddm(d, sigma, trialCondition[0], trialCondition[1])
            except:
                print("An exception occurred while running the model for "
                      "likelihood computation, at simulation " + str(sim) + ".")
                raise
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
                        help="Size of the thread pool.")
    parser.add_argument("--num-values", type=int, default=4,
                        help="Number of item values to use in the artificial "
                        "data.")
    parser.add_argument("--num-trials", type=int, default=10,
                        help="Number of artificial data trials to be generated "
                        "per trial condition.")
    parser.add_argument("--num-simulations", type=int, default=10,
                        help="Number of simulations to be generated per trial "
                        "condition, to be used in the RT histograms.")
    parser.add_argument("--bin-step", type=int, default=100,
                        help="Size of the bin step to be used in the RT "
                        "histograms.")
    parser.add_argument("--max-rt", type=int, default=8000,
                        help="Maximum RT to be used in the RT histograms.")
    parser.add_argument("--d", type=float, default=0.006,
                        help="DDM parameter for generating artificial data.")
    parser.add_argument("--sigma", type=float, default=0.08,
                        help="DDM parameter for generating artificial data.")
    parser.add_argument("--range-d", nargs="+", type=float,
                        default=[0.005, 0.006, 0.007],
                        help="Search range for parameter d.")
    parser.add_argument("--range-sigma", nargs="+", type=float,
                        default=[0.065, 0.08, 0.095],
                        help="Search range for parameter sigma.")
    parser.add_argument("--verbose", default=False, action="store_true",
                        help="Increase output verbosity.")
    args = parser.parse_args()

    pool = Pool(args.num_threads)

    histBins = range(0, args.max_rt + args.bin_step, args.bin_step)

    values = range(1, args.num_values + 1, 1)
    trialConditions = list()
    for vLeft in values:
        for vRight in values:
            trialConditions.append((vLeft, vRight))

    # Generate artificial data.
    dataRTLeft = dict()
    dataRTRight = dict()
    for trialCondition in trialConditions:
        dataRTLeft[trialCondition] = list()
        dataRTRight[trialCondition] = list()
    for trialCondition in trialConditions:
        trial = 0
        while trial < args.num_trials:
            try:
                results = ddm(args.d, args.sigma, trialCondition[0],
                              trialCondition[1])
            except Exception as e:
                print("An exception occurred while running the model for "
                      "artificial data generation, at trial " + str(trial) +
                      ": " + str(e))
                return
            RT = results.RT
            choice = results.choice
            if choice == -1:
                dataRTLeft[trialCondition].append(RT)
            elif choice == 1:
                dataRTRight[trialCondition].append(RT)
            trial += 1

    # Generate histograms for artificial data.
    dataHistLeft = dict()
    dataHistRight = dict()
    for trialCondition in trialConditions:
        dataHistLeft[trialCondition] = np.histogram(
            dataRTLeft[trialCondition], bins=histBins)[0]
        dataHistRight[trialCondition] = np.histogram(
            dataRTRight[trialCondition], bins=histBins)[0]

    # Grid search on the parameters of the model.
    models = list()
    for d in args.range_d:
        for sigma in args.range_sigma:
            model = (d, sigma)
            models.append(model)

    listParams = list()
    for model in models:
        listParams.append(
            (model[0], model[1], trialConditions, args.num_simulations,
            histBins, dataHistLeft, dataHistRight))
    likelihoods = pool.map(get_model_likelihood_wrapper, listParams)

    if args.verbose:
        for i in xrange(len(models)):
            print("L" + str(models[i]) + " = " + str(likelihoods[i]))
        bestIndex = likelihoods.index(max(likelihoods))
        print("Best fit: " + str(models[bestIndex]))


if __name__ == '__main__':
    main()
