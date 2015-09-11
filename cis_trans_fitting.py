#!/usr/bin/python

"""
cis_trans_fitting.py
Author: Gabriela Tavares, gtavares@caltech.edu

Maximum likelihood estimation procedure for the attentional drift-diffusion
model (aDDM), specific for perceptual decisions, allowing for analysis of cis
trials or trans trials exclusively. A grid search is performed over the 3 free
parameters of the model. Data from all subjects is pooled such that a single set
of optimal parameters is estimated. aDDM simulations are generated for the model
estimated.
"""

from multiprocessing import Pool

import argparse
import numpy as np
import sys

from addm import (get_trial_likelihood, get_empirical_distributions,
                  run_simulations)
from util import load_data_from_csv, save_simulations_to_csv


def get_model_nll(choice, valueLeft, valueRight, fixItem, fixTime, d, theta,
                  sigma, trialsPerSubject=100, useOddTrials=True,
                  useEvenTrials=True, isCisTrial=None, isTransTrial=None,
                  useCisTrials=True, useTransTrials=True, verbose=True):
    """
    Computes the negative log likelihood of a data set given the parameters of
    the aDDM.
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
      trialsPerSubject: integer, number of trials to be used from each subject.
          If smaller than one, all trials will be used.
      useOddTrials: boolean, whether or not to use odd trials in the analysis.
      useEvenTrials: boolean, whether or not to use even trials in the analysis.
      isCisTrial: dict of dicts, indexed first by subject then by trial number.
          Each entry is a boolean indicating if the trial is cis (both bars on
          the same side of the target).
      isTransTrial: dict of dicts with same indexing as isCisTrial. Each entry
          is a boolean indicating if the trial is trans (bars on either side of
          the target).
      useCisTrials: boolean, whether or not to use cis trials  in the analysis.
      useTransTrials: boolean, whether or not to use trans trials in the
          analysis.
      verbose: boolean, whether or not to print updates during computation.
    Returns:
      The negative log likelihood for the given data set and model.
    """

    logLikelihood = 0
    subjects = choice.keys()
    for subject in subjects:
        trials = choice[subject].keys()
        if trialsPerSubject < 1:
            trialsPerSubject = len(trials)

        if useEvenTrials and useOddTrials:
            if useCisTrials and useTransTrials:
                trialSet = np.random.choice(trials, trialsPerSubject,
                                            replace=False)
            elif useCisTrials and not useTransTrials:
                trialSet = np.random.choice(
                    [trial for trial in trials if isCisTrial[subject][trial]],
                    trialsPerSubject, replace=False)
            elif not useCisTrials and useTransTrials:
                trialSet = np.random.choice(
                    [trial for trial in trials if isTransTrial[subject][trial]],
                    trialsPerSubject, replace=False)
            else:
                return 0
        elif useEvenTrials and not useOddTrials:
            if useCisTrials and useTransTrials:
                trialSet = np.random.choice(
                    [trial for trial in trials if not trial % 2],
                    trialsPerSubject, replace=False)
            elif useCisTrials and not useTransTrials:
                trialSet = np.random.choice(
                    [trial for trial in trials if not trial % 2 and
                     isCisTrial[subject][trial]],
                    trialsPerSubject, replace=False)
            elif not useCisTrials and useTransTrials:
                trialSet = np.random.choice(
                    [trial for trial in trials if not trial % 2 and
                     isTransTrial[subject][trial]],
                    trialsPerSubject, replace=False)
            else:
                return 0
        elif not useEvenTrials and useOddTrials:
            if useCisTrials and useTransTrials:
                trialSet = np.random.choice(
                    [trial for trial in trials if trial % 2],
                    trialsPerSubject, replace=False)
            elif useCisTrials and not useTransTrials:
                trialSet = np.random.choice(
                    [trial for trial in trials if trial % 2 and
                     isCisTrial[subject][trial]],
                    trialsPerSubject, replace=False)
            elif not useCisTrials and useTransTrials:
                trialSet = np.random.choice(
                    [trial for trial in trials if trial % 2 and
                     isTransTrial[subject][trial]],
                    trialsPerSubject, replace=False)
            else:
                return 0
        else:
            return 0

        for trial in trialSet:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-threads", type=int, default=9,
                        help="Size of the thread pool.")
    parser.add_argument("--trials-per-subject", type=int, default=100,
                        help="Number of trials from each subject to be used in "
                        "the analysis; if smaller than 1, all trials are used.")
    parser.add_argument("--num-simulations", type=int, default=400,
                        help="Number of simulations to be generated per trial "
                        "condition.")
    parser.add_argument("--range-d", nargs="+", type=float,
                        default=[0.003, 0.006, 0.009],
                        help="Search range for parameter d.")
    parser.add_argument("--range-sigma", nargs="+", type=float,
                        default=[0.03, 0.06, 0.09],
                        help="Search range for parameter sigma.")
    parser.add_argument("--range-theta", nargs="+", type=float,
                        default=[0.3, 0.5, 0.7],
                        help="Search range for parameter theta.")
    parser.add_argument("--expdata-file-name", type=str, default="expdata.csv",
                        help="Name of experimental data file.")
    parser.add_argument("--fixations-file-name", type=str,
                        default="fixations.csv", help="Name of fixations file.")
    parser.add_argument("--use-cis-trials", default=False, action="store_true",
                        help="Use CIS trials in the analysis.")
    parser.add_argument("--use-trans-trials", default=False,
                        action="store_true", help="Use TRANS trials in the "
                        "analysis.")
    parser.add_argument("--save-simulations", default=False,
                        action="store_true", help="Save simulations to CSV.")
    parser.add_argument("--verbose", default=False, action="store_true",
                        help="Increase output verbosity.")
    args = parser.parse_args()

    pool = Pool(args.num_threads)

    # Load experimental data from CSV file.
    data = load_data_from_csv(args.expdata_file_name, args.fixations_file_name,
                              useAngularDists=True)
    choice = data.choice
    valueLeft = data.valueLeft
    valueRight = data.valueRight
    fixItem = data.fixItem
    fixTime = data.fixTime
    isCisTrial = data.isCisTrial
    isTransTrial = data.isTransTrial

    # Maximum likelihood estimation.
    # Grid search on the parameters of the model using odd trials only.
    if args.verbose:
        print("Starting grid search...")
    models = list()
    listParams = list()
    for d in args.range_d:
        for theta in args.range_theta:
            for sigma in args.range_sigma:
                models.append((d, theta, sigma))
                params = (choice, valueLeft, valueRight, fixItem, fixTime, d,
                          theta, sigma, args.trials_per_subject, True, False,
                          isCisTrial, isTransTrial, args.use_cis_trials,
                          args.use_trans_trials, args.verbose)
                listParams.append(params)
    results = pool.map(get_model_nll_wrapper, listParams)

    # Get optimal parameters.
    minNegLogLikeIdx = results.index(min(results))
    optimD = models[minNegLogLikeIdx][0]
    optimTheta = models[minNegLogLikeIdx][1]
    optimSigma = models[minNegLogLikeIdx][2]
    if args.verbose:
        print("Finished grid search!")
        print("Optimal d: " + str(optimD))
        print("Optimal theta: " + str(optimTheta))
        print("Optimal sigma: " + str(optimSigma))
        print("Min NLL: " + str(min(results)))

    # Get empirical distributions from even trials only.
    evenDists = get_empirical_distributions(
        valueLeft, valueRight, fixItem, fixTime, useOddTrials=False,
        useEvenTrials=True, isCisTrial=isCisTrial, isTransTrial=isTransTrial,
        useCisTrials=args.use_cis_trials, useTransTrials=args.use_trans_trials)
    probLeftFixFirst = evenDists.probLeftFixFirst
    distLatencies = evenDists.distLatencies
    distTransitions = evenDists.distTransitions
    distFixations = evenDists.distFixations

    # Parameters for generating simulations.
    orientations = range(-15,20,5)
    trialConditions = list()
    for oLeft in orientations:
        for oRight in orientations:
            vLeft = np.absolute((np.absolute(oLeft) - 15) / 5)
            vRight = np.absolute((np.absolute(oRight) - 15) / 5)
            if oLeft != oRight and args.use_cis_trials and oLeft * oRight >= 0:
                trialConditions.append((vLeft, vRight))
            elif (oLeft != oRight and args.use_trans_trials and
                  oLeft * oRight <= 0):
                trialConditions.append((vLeft, vRight))

    # Generate simulations using the empirical distributions and the
    # estimated parameters.
    simul = run_simulations(
        probLeftFixFirst, distLatencies, distTransitions, distFixations,
        args.num_simulations, trialConditions, optimD, optimTheta,
        sigma=optimSigma)
    simulRT = simul.RT
    simulChoice = simul.choice
    simulValueLeft = simul.valueLeft
    simulValueRight = simul.valueRight
    simulFixItem = simul.fixItem
    simulFixTime = simul.fixTime
    simulFixRDV = simul.fixRDV

    if args.save_simulations:
        totalTrials = args.num_simulations * len(trialConditions)
        save_simulations_to_csv(
            simulChoice, simulRT, simulValueLeft, simulValueRight, simulFixItem,
            simulFixTime, simulFixRDV, totalTrials)


if __name__ == '__main__':
    main()
