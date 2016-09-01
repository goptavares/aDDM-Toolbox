#!/usr/bin/python

"""
addm_test.py
Author: Gabriela Tavares, gtavares@caltech.edu

Test to check the validity of the addm parameter estimation. Artificil data is
generated using specific parameters for the model. Fixations are sampled from
the data pooled from all subjects (or from a subset of subjects, when
provided). The parameters used for data generation are then recovered through a
posterior distribution estimation procedure.
"""

import argparse
import numpy as np

from addm import aDDM
from util import load_data_from_csv, get_empirical_distributions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-ids", nargs="+", type=str, default=[],
                        help="List of subject ids. If not provided, all "
                        "existing subjects will be used.")
    parser.add_argument("--num-threads", type=int, default=9,
                        help="Size of the thread pool.")
    parser.add_argument("--num-trials", type=int, default=800,
                        help="Number of artificial data trials to be "
                        "generated per trial condition.")
    parser.add_argument("--d", type=float, default=0.006,
                        help="aDDM parameter for generating artificial data.")
    parser.add_argument("--sigma", type=float, default=0.08,
                        help="aDDM parameter for generating artificial data.")
    parser.add_argument("--theta", type=float, default=0.5,
                        help="aDDM parameter for generating artificial data.")
    parser.add_argument("--range-d", nargs="+", type=float,
                        default=[0.005, 0.006, 0.007],
                        help="Search range for parameter d.")
    parser.add_argument("--range-sigma", nargs="+", type=float,
                        default=[0.065, 0.08, 0.095],
                        help="Search range for parameter sigma.")
    parser.add_argument("--range-theta", nargs="+", type=float,
                        default=[0.4, 0.5, 0.6],
                        help="Search range for parameter theta.")
    parser.add_argument("--expdata-file-name", type=str, default="expdata.csv",
                        help="Name of experimental data file.")
    parser.add_argument("--fixations-file-name", type=str,
                        default="fixations.csv",
                        help="Name of fixations file.")
    parser.add_argument("--verbose", default=False, action="store_true",
                        help="Increase output verbosity.")
    args = parser.parse_args()

    # Load experimental data from CSV file.
    if args.verbose:
        print("Loading experimental data...")
    try:
        data = load_data_from_csv(
            args.expdata_file_name, args.fixations_file_name,
            useAngularDists=True)
    except:
        print("An exception occurred while loading the data.")
        raise

    # Get fixation distributions.
    subjectIds = args.subject_ids if args.subject_ids else None
    try:
        fixationData = get_empirical_distributions(data, subjectIds=subjectIds)
    except:
        print("An exception occurred while getting fixation distributions.")
        raise

    # Generate artificial data.
    if args.verbose:
        print("Generating artificial data...")
    model = aDDM(args.d, args.sigma, args.theta)
    trials = list()
    orientations = range(-15, 20, 5)
    for orLeft in orientations:
        for orRight in orientations:
            if orLeft == orRight:
                continue
            valueLeft = np.absolute((np.absolute(orLeft) - 15) / 5)
            valueRight = np.absolute((np.absolute(orRight) - 15) / 5)
            for t in xrange(args.num_trials):
                try:
                    trials.append(
                        model.simulate_trial(valueLeft, valueRight,
                                             fixationData))
                except:
                    print("An exception occurred while generating artificial "
                          "data.")
                    raise

    # Get likelihoods for all models and all artificial trials.
    numModels = (len(args.range_d) * len(args.range_sigma) *
                 len(args.range_theta))
    likelihoods = dict()
    models = list()
    posteriors = dict()
    for d in args.range_d:
        for sigma in args.range_sigma:
            for theta in args.range_theta:
                model = aDDM(d, sigma, theta)
                if args.verbose:
                    print("Computing likelihoods for model " +
                          str(model.params) + "...")
                try:
                    likelihoods[model.params] = model.parallel_get_likelihoods(
                        trials, numThreads=args.num_threads)
                except:
                    print("An exception occurred during the likelihood "
                          "computations for model " + str(model.params) + ".")
                    raise
                models.append(model)
                posteriors[model.params] = 1. / numModels

    # Compute the posteriors.
    for t in xrange(len(trials)):
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
            posteriors[model.params] = (likelihoods[model.params][t] *
                                        prior / denominator)

    if args.verbose:
        for model in models:
            print("P" + str(model.params) +  " = " +
                  str(posteriors[model.params]))
        print("Sum: " + str(sum(posteriors.values())))


if __name__ == '__main__':
    main()
