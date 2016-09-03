#!/usr/bin/python

"""
addm_posteriors.py
Author: Gabriela Tavares, gtavares@caltech.edu

Posterior distribution estimation procedure for the attentional drift-diffusion
model (aDDM), using a grid search over the 3 free parameters of the model. Data
from all subjects is pooled such that a single set of optimal parameters is
estimated (or from a subset of subjects, when provided).

aDDM simulations are generated according to the posterior distribution obtained
(instead of generating simulations from a single model, we sample models from
the posterior distribution and simulate them, then aggregate all simulations).
"""

import matplotlib
matplotlib.use('Agg')

from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages

import argparse
import numpy as np

from addm import aDDM
from util import (load_data_from_csv, get_empirical_distributions,
                  save_simulations_to_csv, generate_choice_curves,
                  generate_rt_curves)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-ids", nargs="+", type=str, default=[],
                        help="List of subject ids. If not provided, all "
                        "existing subjects will be used.")
    parser.add_argument("--num-threads", type=int, default=9,
                        help="Size of the thread pool.")
    parser.add_argument("--trials-per-subject", type=int, default=100,
                        help="Number of trials from each subject to be used "
                        "in the analysis; if smaller than 1, all trials are "
                        "used.")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of samples to be drawn from the "
                        "posterior distribution when generating simulations.")
    parser.add_argument("--num-simulations-per-sample", type=int, default=10,
                        help="Number of simulations to be genearated for each "
                        "sample drawn from the posterior distribution.")
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
                        default="fixations.csv",
                        help="Name of fixations file.")
    parser.add_argument("--save-simulations", default=False,
                        action="store_true", help="Save simulations to CSV.")
    parser.add_argument("--save-figures", default=False,
                        action="store_true", help="Save figures comparing "
                        "choice and RT curves for data and simulations.")
    parser.add_argument("--verbose", default=False, action="store_true",
                        help="Increase output verbosity.")
    args = parser.parse_args()

    # Load experimental data from CSV file.
    if args.verbose:
        print("Loading experimental data...")
    data = load_data_from_csv(
        args.expdata_file_name, args.fixations_file_name, useAngularDists=True)

    # Begin posterior estimation using odd trials only.
    if args.verbose:
        print("Starting grid search...")

    # Get correct subset of trials.
    dataTrials = list()
    subjectIds = args.subject_ids if args.subject_ids else data.keys()
    for subjectId in subjectIds:
        numTrials = (args.trials_per_subject if args.trials_per_subject >= 1
                     else len(data[subjectId]))
        trialSet = np.random.choice(
            [trialId for trialId in range(len(data[subjectId]))
             if trialId % 2],
            numTrials, replace=False)
        dataTrials.extend([data[subjectId][t] for t in trialSet])

    # Create all models to be used in the grid search.
    numModels = (len(args.range_d) * len(args.range_theta) *
                 len(args.range_sigma))
    models = list()
    posteriors = dict()
    for d in args.range_d:
        for sigma in args.range_sigma:
            for theta in args.range_theta:
                model = aDDM(d, sigma, theta)
                models.append(model)
                posteriors[model.params] = 1. / numModels

    # Get likelihoods for all models.
    likelihoods = dict()
    for model in models:
        if args.verbose:
            print("Computing likelihoods for model " +
                  str(model.params) + "...")
        try:
            likelihoods[model.params] = model.parallel_get_likelihoods(
                dataTrials, numThreads=args.num_threads)
        except:
            print("An exception occurred during the likelihood " +
                  "computations for model " + str(model.params) + ".")
            raise

    if args.verbose:
        print("Finished grid search!")

    # Compute posterior distribution over all models.
    for t in xrange(len(dataTrials)):
        # Get the denominator for normalizing the posteriors.
        denominator = 0
        for model in models:
            denominator += (posteriors[model.params] *
                            likelihoods[model.params][t])
        if denominator == 0:
            continue

        # Calculate the posteriors after this trial.
        for model in models:
            prior = posteriors[model.params]
            posteriors[model.params] = (likelihoods[model.params][t] * prior /
                                        denominator)

    # Get fixation distributions from even trials.
    if args.verbose:
        print("Getting fixation distributions from even trials...")
    fixationData = get_empirical_distributions(
        data, subjectIds=subjectIds, useOddTrials=False, useEvenTrials=True)

    # Get list of posterior distribution values.
    posteriorsList = list()
    for model in models:
        posteriorsList.append(posteriors[model.params])

    # Generate probabilistic set of simulations using the posterior
    # distribution.
    simulTrials = list()
    orientations = range(-15,20,5)
    for orLeft in orientations:
        for orRight in orientations:
            if orLeft == orRight:
                continue
            valueLeft = np.absolute((np.absolute(orLeft) - 15) / 5)
            valueRight = np.absolute((np.absolute(orRight) - 15) / 5)
            for s in xrange(args.num_samples):
                # Sample model from posteriors distribution.
                modelIndex = np.random.choice(
                    np.array(range(numModels)), p=np.array(posteriorsList))
                model = models[modelIndex]
                for t in xrange(args.num_simulations_per_sample):
                    try:
                        simulTrials.append(
                            model.simulate_trial(valueLeft, valueRight,
                                                 fixationData))
                    except:
                        print("An exception occurred while generating " +
                              "artificial trial " + str(t) + " for " +
                              "condition " + str(valueLeft) + ", " +
                              str(valueRight) + " and model " +
                              str(model.params) + " (sample " + str(s) + ").")
                        raise

    if args.save_simulations:
        save_simulations_to_csv(simulTrials)

    if args.save_figures:
        # Create pdf file to save figures.
        currTime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        pp = PdfPages("addm_fit_" + currTime + ".pdf")

        # Generate choice and RT curves for real data (odd trials) and
        # simulations (generated from even trials).
        fig1 = generate_choice_curves(dataTrials, simulTrials)
        pp.savefig(fig1)
        fig2 = generate_rt_curves(dataTrials, simulTrials)
        pp.savefig(fig2)
        pp.close()


if __name__ == '__main__':
    main()
