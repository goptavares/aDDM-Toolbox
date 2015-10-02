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

import argparse
import numpy as np

from addm import (get_trial_likelihood, get_empirical_distributions,
                  run_simulations)
from util import (load_data_from_csv, save_simulations_to_csv,
                  generate_choice_curves, generate_rt_curves)


def get_model_nll(choice, valueLeft, valueRight, fixItem, fixTime, d, theta,
                  sigma, numTrials=100, useOddTrials=True, useEvenTrials=True,
                  verbose=True):
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
      numTrials: integer, number of trials to be used from the subject's data.
          If smaller than one, all trials will be used.
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
        trials = choice[subject].keys()
        if numTrials < 1:
            numTrials = len(trials)
        if useEvenTrials and useOddTrials:
            trialSet = np.random.choice(trials, numTrials, replace=False)
        elif useEvenTrials and not useOddTrials:
            trialSet = np.random.choice(
                [trial for trial in trials if not trial % 2],
                numTrials, replace=False)
        elif not useEvenTrials and useOddTrials:
            trialSet = np.random.choice(
                [trial for trial in trials if trial % 2],
                numTrials, replace=False)
        else:
            return 0

        for trial in trialSet:
            if not useOddTrials and trial % 2 != 0:
                continue
            if not useEvenTrials and trial % 2 == 0:
                continue
            try:
                likelihood = get_trial_likelihood(
                    choice[subject][trial], valueLeft[subject][trial],
                    valueRight[subject][trial], fixItem[subject][trial],
                    fixTime[subject][trial], d, theta, sigma=sigma)
            except:
                print("An exception occurred during the likelihood computation "
                      "for trial " + str(trial) + ".")
                raise
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
    parser.add_argument("subject", type=str, help="Subject name")
    parser.add_argument("--num-threads", type=int, default=9,
                        help="Size of the thread pool.")
    parser.add_argument("--num-trials", type=int, default=100,
                        help="Number of trials to be used in the analysis; if "
                        "smaller than 1, all trials are used.")
    parser.add_argument("--num-simulations", type=int, default=32,
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
    parser.add_argument("--save-simulations", default=False,
                        action="store_true", help="Save simulations to CSV.")
    parser.add_argument("--save-figures", default=False,
                        action="store_true", help="Save figures comparing "
                        "choice and RT curves for data and simulations.")
    parser.add_argument("--verbose", default=False, action="store_true",
                        help="Increase output verbosity.")
    args = parser.parse_args()

    pool = Pool(args.num_threads)

    choice = dict()
    valueLeft = dict()
    valueRight = dict()
    fixItem = dict()
    fixTime = dict()

    # Load experimental data from CSV file.
    try:
        data = load_data_from_csv(
            args.expdata_file_name, args.fixations_file_name,
            useAngularDists=True)
    except Exception as e:
        print("An exception occurred while loading the data: " + str(e))
        return
    choice[args.subject] = data.choice[args.subject]
    valueLeft[args.subject] = data.valueLeft[args.subject]
    valueRight[args.subject] = data.valueRight[args.subject]
    fixItem[args.subject] = data.fixItem[args.subject]
    fixTime[args.subject] = data.fixTime[args.subject]

    # Maximum likelihood estimation using odd trials only.
    # Grid search on the parameters of the model.
    if args.verbose:
        print("Starting grid search for subject " + args.subject + "...")
    models = list()
    listParams = list()
    for d in args.range_d:
        for theta in args.range_theta:
            for sigma in args.range_sigma:
                models.append((d, theta, sigma))
                params = (choice, valueLeft, valueRight, fixItem, fixTime, d,
                          theta, sigma, args.num_trials, True, False,
                          args.verbose)
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

    # Get empirical distributions from even trials.
    try:
        dists = get_empirical_distributions(
            valueLeft, valueRight, fixItem, fixTime, useOddTrials=False,
            useEvenTrials=True)
    except Exception as e:
        print("An exception occurred while getting empirical distributions: " +
              str(e))
        return
    probLeftFixFirst = dists.probLeftFixFirst
    distLatencies = dists.distLatencies
    distTransitions = dists.distTransitions
    distFixations = dists.distFixations

    # Trial conditions for generating simulations.
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
    try:
        simul = run_simulations(
            probLeftFixFirst, distLatencies, distTransitions, distFixations,
            args.num_simulations, trialConditions, optimD, optimTheta,
            sigma=optimSigma)
    except Exception as e:
        print("An exception occurred while running simulations: " + str(e))
        return
    simulRT = simul.RT
    simulChoice = simul.choice
    simulValueLeft = simul.valueLeft
    simulValueRight = simul.valueRight
    simulFixItem = simul.fixItem
    simulFixTime = simul.fixTime
    simulFixRDV = simul.fixRDV

    totalTrials = args.num_simulations * len(trialConditions)

    if args.save_simulations:
        save_simulations_to_csv(
            simulChoice, simulRT, simulValueLeft, simulValueRight, simulFixItem,
            simulFixTime, simulFixRDV, totalTrials)

    if args.save_figures:
        # Create pdf file to save figures.
        pp = PdfPages(
            "figures_" + str(optimD) + "_" + str(optimTheta) + "_" +
            str(optimSigma) + "_" + str(args.num_simulations) + ".pdf")

        # Generate choice and RT curves for real data (odd trials) and
        # simulations (generated from even trials).
        fig1 = generate_choice_curves(
            choice, valueLeft, valueRight, simulChoice, simulValueLeft,
            simulValueRight, totalTrials)
        pp.savefig(fig1)
        fig2 = generate_rt_curves(
            RT, valueLeft, valueRight, simulRT, simulValueLeft, simulValueRight,
            totalTrials)
        pp.savefig(fig2)
        pp.close()


if __name__ == '__main__':
    main()
