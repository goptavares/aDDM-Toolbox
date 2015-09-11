#!/usr/bin/python

"""
ddm_test.py
Author: Gabriela Tavares, gtavares@caltech.edu

Test to check the validity of the ddm parameter estimation. Artificil data is
generated using specific parameters for the model. These parameters are then
recovered through a posterior distribution estimation procedure.
"""

from multiprocessing import Pool

import argparse

from ddm import get_trial_likelihood, run_simulations


def get_trial_likelihood_wrapper(params):
    """
    Wrapper for ddm.get_trial_likelihood() which takes a single argument.
    Intended for parallel computation using a thread pool.
    Args:
      params: tuple consisting of all arguments required by
          ddm.get_trial_likelihood().
    Returns:
      The output of ddm.get_trial_likelihood().
    """

    return get_trial_likelihood(*params)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-threads", type=int, default=9,
                        help="Size of the thread pool.")
    parser.add_argument("--num-values", type=int, default=4,
                        help="Number of item values to use in the artificial "
                        "data.")
    parser.add_argument("--num-trials", type=int, default=32,
                        help="Number of artificial data trials to be generated "
                        "per trial condition.")
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

    # Trial conditions for generating artificial data.
    values = range(1, args.num_values + 1, 1)
    trialConditions = list()
    for vLeft in values:
        for vRight in values:
            trialConditions.append((vLeft, vRight))

    # Generate artificial data.
    simul = run_simulations(args.num_trials, trialConditions, args.d,
                            args.sigma)
    RT = simul.RT
    choice = simul.choice
    valueLeft = simul.valueLeft
    valueRight = simul.valueRight

    numModels = len(args.range_d) * len(args.range_sigma)
    models = list()
    posteriors = dict()
    for d in args.range_d:
        for sigma in args.range_sigma:
            model = (d, sigma)
            models.append(model)
            posteriors[model] = 1./ numModels

    trials = RT.keys()
    for trial in trials:
        listParams = list()
        for model in models:
            listParams.append(
                (RT[trial], choice[trial], valueLeft[trial], valueRight[trial],
                model[0], model[1]))
        likelihoods = pool.map(get_trial_likelihood_wrapper, listParams)

        # Get the denominator for normalizing the posteriors.
        i = 0
        denominator = 0
        for model in models:
            denominator += posteriors[model] * likelihoods[i]
            i += 1
        if denominator == 0:
            continue

        # Calculate the posteriors after this trial.
        i = 0
        for model in models:
            prior = posteriors[model]
            posteriors[model] = likelihoods[i] * prior / denominator
            i += 1

        if args.verbose and trial % 200 == 0:
            for model in posteriors:
                print("P" + str(model) + " = " + str(posteriors[model]))
            print("Sum: " + str(sum(posteriors.values())))

    if args.verbose:
        for model in posteriors:
            print("P" + str(model) + " = " + str(posteriors[model]))
        print("Sum: " + str(sum(posteriors.values())))


if __name__ == '__main__':
    main()
