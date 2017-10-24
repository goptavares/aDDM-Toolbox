#!/usr/bin/python

"""
Copyright (C) 2017, California Institute of Technology

This file is part of addm_toolbox.

addm_toolbox is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

addm_toolbox is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with addm_toolbox. If not, see <http://www.gnu.org/licenses/>.

---

Module: basinhopping_optimize.py
Author: Gabriela Tavares, gtavares@caltech.edu

Maximum likelihood estimation procedure for the attentional drift-diffusion
model (aDDM), using a Basinhopping algorithm to search the parameter space.
Data from all subjects is pooled such that a single set of optimal parameters
is estimated.
"""

from __future__ import division, absolute_import

import argparse
import numpy as np
import os
import sys

from builtins import range, str
from scipy.optimize import basinhopping

from addm_toolbox.addm import aDDM
from addm_toolbox.util import load_data_from_csv, convert_item_values


# Global variables.
dataTrials = []


def get_model_nll(params):
    """
    Computes the negative log likelihood of the global data set given the
    parameters of the aDDM.
    Args:
      params: list containing the 3 model parameters, in the following order:
          d, theta, sigma.
    Returns:
      The negative log likelihood for the global data set and the given model.
    """
    d = params[0]
    theta = params[1]
    sigma = params[2]
    model = aDDM(d, sigma, theta) 

    logLikelihood = 0
    for trial in dataTrials:
        try:
            likelihood = model.get_trial_likelihood(trial)
        except:
            print(u"An exception occurred during the likelihood " +
                  "computations for model " + str(model.params) + u".")
            raise
        if likelihood != 0:
            logLikelihood += np.log(likelihood)

    print(u"NLL for " + str(params) + u": " + str(-logLikelihood))
    if logLikelihood != 0:
        return -logLikelihood
    else:
        return sys.maxint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(u"--subject-ids", nargs=u"+", type=str, default=[],
                        help=u"List of subject ids. If not provided, all "
                        "existing subjects will be used.")
    parser.add_argument(u"--trials-per-subject", type=int, default=100,
                        help=u"Number of trials from each subject to be used "
                        "in the analysis; if smaller than 1, all trials are "
                        "used.")
    parser.add_argument(u"--num-iterations", type=int, default=100,
                        help=u"Number of basin hopping iterations.")
    parser.add_argument(u"--step-size", type=float, default=0.001,
                        help=u"Step size for use in the random displacement "
                        "of the basin hopping algorithm.")
    parser.add_argument(u"--initial-d", type=float, default=0.005,
                        help=u"Initial value for parameter d.")
    parser.add_argument(u"--initial-theta", type=float, default=0.5,
                        help=u"Initial value for parameter theta.")
    parser.add_argument(u"--initial-sigma", type=float, default=0.05,
                        help=u"Initial value for parameter sigma.")
    parser.add_argument(u"--lower-bound-d", type=float, default=0.0001,
                        help=u"Lower search bound for parameter d.")
    parser.add_argument(u"--upper-bound-d", type=float, default=0.01,
                        help=u"Upper search bound for parameter d.")
    parser.add_argument(u"--lower-bound-theta", type=float, default=0,
                        help=u"Lower search bound for parameter theta.")
    parser.add_argument(u"--upper-bound-theta", type=float, default=1,
                        help=u"Upper search bound for parameter theta.")
    parser.add_argument(u"--lower-bound-sigma", type=float, default=0.001,
                        help=u"Lower search bound for parameter sigma.")
    parser.add_argument(u"--upper-bound-sigma", type=float, default=0.1,
                        help=u"Upper search bound for parameter sigma.")
    parser.add_argument(u"--expdata-file-name", type=str,
                        default=os.path.join(os.path.dirname(
                            os.path.realpath(__file__)),
                            u"addm_toolbox/data/expdata.csv"),
                        help=u"Name of experimental data file.")
    parser.add_argument(u"--fixations-file-name", type=str,
                        default=os.path.join(os.path.dirname(
                            os.path.realpath(__file__)),
                            u"addm_toolbox/data/fixations.csv"),
                        help=u"Name of fixations file.")
    parser.add_argument(u"--verbose", default=False, action=u"store_true",
                        help=u"Increase output verbosity.")
    args = parser.parse_args()

    global dataTrials

    # Load experimental data from CSV file.
    if args.verbose:
        print(u"Loading experimental data...")
    data = load_data_from_csv(
        args.expdata_file_name, args.fixations_file_name,
        convertItemValues=convert_item_values)

    # Get correct subset of trials.
    subjectIds = args.subject_ids if args.subject_ids else list(data)
    for subjectId in subjectIds:
        numTrials = (args.trials_per_subject if args.trials_per_subject >= 1
                     else len(data[subjectId]))
        trialSet = np.random.choice(
            [trialId for trialId in range(len(data[subjectId]))],
            numTrials, replace=False)
        dataTrials.extend([data[subjectId][t] for t in trialSet])

    # Initial guess for the parameters: d, theta, sigma.
    initialParams = [args.initial_d, args.initial_theta, args.initial_sigma]

    # Search bounds.
    bounds = [(args.lower_bound_d, args.upper_bound_d),
              (args.lower_bound_theta, args.upper_bound_theta),
              (args.lower_bound_sigma, args.upper_bound_sigma)
             ]

    # Optimize using Basinhopping algorithm.
    minimizerKwargs = dict(method=u"L-BFGS-B", bounds=bounds)
    result = basinhopping(
        get_model_nll, initialParams, minimizer_kwargs=minimizerKwargs,
        niter=args.num_iterations,stepsize=args.step_size)
    print(u"Optimization result: " + str(result))


if __name__ == u"__main__":
    main()
