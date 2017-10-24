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

Module: ddm_mla_test.py
Author: Gabriela Tavares, gtavares@caltech.edu

This script performs a test to check the validity of the maximum likelihood
algorithm (MLA) for the drift-diffusion model (DDM). Artificial data is
generated using specific parameters for the model. These parameters are then
recovered through a maximum likelihood estimation procedure, using a grid
search over the 2 free parameters of the model.
"""

from __future__ import absolute_import

import argparse
import numpy as np
import os

from builtins import range, str
from multiprocessing import Pool

from addm_toolbox.ddm_mla import DDM
from addm_toolbox.util import load_trial_conditions_from_csv


def wrap_ddm_get_model_log_likelihood(args):
    """
    Wrapper for DDM.get_model_log_likelihood(), intended for parallel
    computation using a threadpool.
    Args:
      args: a tuple where the first item is a DDM object, and the remaining
          item are the same arguments required by
          DDM.get_model_log_likelihood().
    Returns:
      The output of DDM.get_model_log_likelihood().
    """
    model = args[0]
    return model.get_model_log_likelihood(*args[1:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(u"--num-threads", type=int, default=9,
                        help=u"Size of the thread pool.")
    parser.add_argument(u"--num-trials", type=int, default=10,
                        help=u"Number of artificial data trials to be "
                        "generated per trial condition.")
    parser.add_argument(u"--num-simulations", type=int, default=10,
                        help=u"Number of simulations to be generated per "
                        "trial condition, to be used in the RT histograms.")
    parser.add_argument(u"--bin-step", type=int, default=100,
                        help=u"Size of the bin step to be used in the RT "
                        "histograms.")
    parser.add_argument(u"--max-rt", type=int, default=8000,
                        help=u"Maximum RT to be used in the RT histograms.")
    parser.add_argument(u"--d", type=float, default=0.006,
                        help=u"DDM parameter for generating artificial data.")
    parser.add_argument(u"--sigma", type=float, default=0.08,
                        help=u"DDM parameter for generating artificial data.")
    parser.add_argument(u"--range-d", nargs=u"+", type=float,
                        default=[0.005, 0.006, 0.007],
                        help=u"Search range for parameter d.")
    parser.add_argument(u"--range-sigma", nargs=u"+", type=float,
                        default=[0.065, 0.08, 0.095],
                        help=u"Search range for parameter sigma.")
    parser.add_argument(u"--trials-file-name", type=str,
                        default=os.path.join(
                            os.path.dirname(os.path.realpath(__file__)),
                            u"addm_toolbox/data/trial_conditions.csv"),
                        help=u"Name of trial conditions file.")
    parser.add_argument(u"--verbose", default=False, action=u"store_true",
                        help=u"Increase output verbosity.")
    args = parser.parse_args()

    pool = Pool(args.num_threads)

    histBins = list(range(0, args.max_rt + args.bin_step, args.bin_step))

    # Load trial conditions.
    trialConditions = load_trial_conditions_from_csv(args.trials_file_name)

    # Generate artificial data.
    dataRTLeft = dict()
    dataRTRight = dict()
    for trialCondition in trialConditions:
        dataRTLeft[trialCondition] = list()
        dataRTRight[trialCondition] = list()
    model = DDM(args.d, args.sigma)
    for trialCondition in trialConditions:
        t = 0
        while t < args.num_trials:
            try:
                trial = model.simulate_trial(trialCondition[0],
                                             trialCondition[1])
            except:
                print(u"An exception occurred while generating artificial "
                      "trial " + str(t) + u" for condition " +
                      str(trialCondition[0]) + u", " + str(trialCondition[1]) +
                      u".")
                raise
            if trial.choice == -1:
                dataRTLeft[trialCondition].append(trial.RT)
            elif trial.choice == 1:
                dataRTRight[trialCondition].append(trial.RT)
            t += 1

    # Generate histograms for artificial data.
    dataHistLeft = dict()
    dataHistRight = dict()
    for trialCondition in trialConditions:
        dataHistLeft[trialCondition] = np.histogram(
            dataRTLeft[trialCondition], bins=histBins)[0]
        dataHistRight[trialCondition] = np.histogram(
            dataRTRight[trialCondition], bins=histBins)[0]

    # Grid search on the parameters of the model.
    if args.verbose:
        print(u"Performing grid search over the model parameters...")
    listParams = list()
    models = list()
    for d in args.range_d:
        for sigma in args.range_sigma:
            model = DDM(d, sigma)
            models.append(model)
            listParams.append((model, trialConditions, args.num_simulations,
                              histBins, dataHistLeft, dataHistRight))
    logLikelihoods = pool.map(wrap_ddm_get_model_log_likelihood, listParams)
    pool.close()

    if args.verbose:
        for i, model in enumerate(models):
            print(u"L" + str(model.params) + u" = " + str(logLikelihoods[i]))
        bestIndex = logLikelihoods.index(max(logLikelihoods))
        print(u"Best fit: " + str(models[bestIndex].params))


if __name__ == u"__main__":
    main()
