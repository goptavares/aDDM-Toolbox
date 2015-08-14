#!/usr/bin/python

"""
individual_mle.py
Author: Gabriela Tavares, gtavares@caltech.edu

Maximum likelihood estimation procedure for the attentional drift-diffusion
model (aDDM), using a grid search over the 3 free parameters of the model, and
using data from a single subject.
"""

import matplotlib
matplotlib.use('Agg')

from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool

import numpy as np

from addm import (get_trial_likelihood, get_empirical_distributions,
                  run_simulations)
from util import (load_data_from_csv, save_simulations_to_csv,
                  generate_choice_curves, generate_rt_curves)


def get_model_nll(choice, valueLeft, valueRight, fixItem, fixTime, d, theta,
                  sigma, useOddTrials=True, useEvenTrials=True, verbose=True):
    """
    Computes the negative log likelihood of a subject's data given the
    parameters of the aDDM.
    Args:
      choice: dict of dicts, indexed first by subject then by trial number. Each
          entry is an integer corresponding to the decision made in that trial.
      valueLeft: dict of dicts with same indexing as choice. Each entry is an
          integer corresponding to the value of the left item.
      valueRight: dict of dicts with same indexing as choice. Each entry is an
          integer corresponding to the value of the right item.
      fixItem: dict of dicts with same indexing as choice. Each entry is an
          ordered list of fixated items in the trial.
      fixTime: dict of dicts with same indexing as choice. Each entry is an
          ordered list of fixation durations in the trial.
      d: float, parameter of the model which controls the speed of integration
          of the signal.
      theta: float between 0 and 1, parameter of the model which controls the
          attentional bias.
      sigma: float, parameter of the model, standard deviation for the normal
          distribution.
      useOddTrials: boolean, whether or not to use odd trials when creating the
          distributions.
      useEvenTrials: boolean, whether or not to use even trials when creating
          the distributions.
      verbose: boolean, whether or not to print updates during computation.
    Returns:
      The negative log likelihood for the given subject and model.
    """

    logLikelihood = 0
    subjects = choice.keys()
    for subject in subjects:
        for trial in choice[subject].keys():
            if not useOddTrials and trial % 2:
                continue
            if not useEvenTrials and not trial % 2:
                continue
            likelihood = get_trial_likelihood(
                choice[subject][trial], valueLeft[subject][trial],
                valueRight[subject][trial], fixItem[subject][trial],
                fixTime[subject][trial], d, theta, sigma=sigma)
            if likelihood != 0:
                logLikelihood += np.log(likelihood)

    if verbose:
        print("NLL for " + str(d) + ", " + str(theta) + ", " + str(sigma) +
              ": " + str(-logLikelihood))
    return -logLikelihood


def get_model_nll_wrapper(params):
    """
    Wrapper for get_model_nll() which takes a single argument. Intended for
    parallel computation using a thread pool.
    Args:
      params: tuple consisting of all arguments required by get_model_nll().
    Returns:
      The output of get_model_nll().
    """

    return get_model_nll(*params)


def main():
    numThreads = 9
    pool = Pool(numThreads)

    subject = "cai"
    choice = dict()
    valueLeft = dict()
    valueRight = dict()
    fixItem = dict()
    fixTime = dict()

    # Load experimental data from CSV file.
    data = load_data_from_csv("expdata.csv", "fixations.csv",
                              useAngularDists=True)
    choice[subject] = data.choice[subject]
    valueLeft[subject] = data.valueLeft[subject]
    valueRight[subject] = data.valueRight[subject]
    fixItem[subject] = data.fixItem[subject]
    fixTime[subject] = data.fixTime[subject]

    # Maximum likelihood estimation using odd trials only.
    # Grid search on the parameters of the model.
    print("Starting grid search for subject " + subject + "...")
    rangeD = [0.003, 0.006, 0.009]
    rangeTheta = [0.2, 0.4, 0.6]
    rangeSigma = [0.06, 0.08, 0.1]

    models = list()
    listParams = list()
    for d in rangeD:
        for theta in rangeTheta:
            for sigma in rangeSigma:
                models.append((d, theta, sigma))
                params = (choice, valueLeft, valueRight, fixItem, fixTime, d,
                          theta, sigma, True, False)
                listParams.append(params)

    results = pool.map(get_model_nll_wrapper, listParams)

    # Get optimal parameters.
    minNegLogLikeIdx = results.index(min(results))
    optimD = models[minNegLogLikeIdx][0]
    optimTheta = models[minNegLogLikeIdx][1]
    optimSigma = models[minNegLogLikeIdx][2]
    print("Finished coarse grid search!")
    print("Optimal d: " + str(optimD))
    print("Optimal theta: " + str(optimTheta))
    print("Optimal sigma: " + str(optimSigma))
    print("Min NLL: " + str(min(results)))

    # Get empirical distributions from even trials.
    evenDists = get_empirical_distributions(
        valueLeft, valueRight, fixItem, fixTime, useOddTrials=False,
        useEvenTrials=True)
    probLeftFixFirst = evenDists.probLeftFixFirst
    distLatencies = evenDists.distLatencies
    distTransitions = evenDists.distTransitions
    distFixations = evenDists.distFixations

    # Parameters for generating simulations.
    numTrials = 32
    orientations = range(-15,20,5)
    trialConditions = list()
    for oLeft in orientations:
        for oRight in orientations:
            if oLeft != oRight:
                vLeft = np.absolute((np.absolute(oLeft) - 15) / 5)
                vRight = np.absolute((np.absolute(oRight) - 15) / 5)
                trialConditions.append((vLeft, vRight))

    # Generate simulations using the even trials distributions and the
    # estimated parameters.
    simul = run_simulations(
        probLeftFixFirst, distLatencies, distTransitions, distFixations,
        numTrials, trialConditions, optimD, optimTheta, sigma=optimSigma)
    simulRT = simul.RT
    simulChoice = simul.choice
    simulValueLeft = simul.valueLeft
    simulValueRight = simul.valueRight
    simulFixItem = simul.fixItem
    simulFixTime = simul.fixTime
    simulFixRDV = simul.fixRDV

    # Create pdf file to save figures.
    pp = PdfPages("figures_" + str(optimD) + "_" + str(optimTheta) + "_" +
                  str(optimSigma) + "_" + str(numTrials) + ".pdf")

    # Generate choice and RT curves for real data (odd trials) and simulations
    # (generated from even trials).
    totalTrials = numTrials * len(trialConditions)
    fig1 = generate_choice_curves(
        choice, valueLeft, valueRight, simulChoice, simulValueLeft,
        simulValueRight, totalTrials)
    pp.savefig(fig1)
    fig2 = generate_rt_curves(
        RT, valueLeft, valueRight, simulRT, simulValueLeft, simulValueRight,
        totalTrials)
    pp.savefig(fig2)
    pp.close()

    save_simulations_to_csv(
        simulChoice, simulRT, simulValueLeft, simulValueRight, simulFixItem,
        simulFixTime, simulFixRDV, totalTrials)


if __name__ == '__main__':
    main()
