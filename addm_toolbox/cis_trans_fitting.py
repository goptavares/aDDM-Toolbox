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

Module: cis_trans_fitting.py
Author: Gabriela Tavares, gtavares@caltech.edu

Maximum likelihood estimation procedure for the attentional drift-diffusion
model (aDDM), specific for perceptual decisions, allowing for analysis of cis
trials or trans trials exclusively. A grid search is performed over the 3 free
parameters of the model. Data from all subjects is pooled such that a single
set of optimal parameters is estimated. aDDM simulations are generated for the
model estimated.
"""

import argparse
import numpy as np
import os
import sys

from datetime import datetime
from multiprocessing import Pool

from addm import aDDM
from util import (load_data_from_csv, get_empirical_distributions,
                  save_simulations_to_csv, generate_choice_curves,
                  generate_rt_curves, convert_item_values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-threads", type=int, default=9,
                        help="Size of the thread pool.")
    parser.add_argument("--subject-ids", nargs="+", type=str, default=[],
                        help="List of subject ids. If not provided, all "
                        "existing subjects will be used.")
    parser.add_argument("--trials-per-subject", type=int, default=100,
                        help="Number of trials from each subject to be used "
                        "in the analysis; if smaller than 1, all trials are "
                        "used.")
    parser.add_argument("--simulations-per-condition", type=int,
                        default=400, help="Number of artificial data trials "
                        "to be generated per trial condition.")
    parser.add_argument("--range-d", nargs="+", type=float,
                        default=[0.003, 0.006, 0.009],
                        help="Search range for parameter d.")
    parser.add_argument("--range-sigma", nargs="+", type=float,
                        default=[0.03, 0.06, 0.09],
                        help="Search range for parameter sigma.")
    parser.add_argument("--range-theta", nargs="+", type=float,
                        default=[0.3, 0.5, 0.7],
                        help="Search range for parameter theta.")
    parser.add_argument("--expdata-file-name", type=str,
                        default=os.path.join(os.path.dirname(
                            os.path.realpath(__file__)), "data/expdata.csv"),
                        help="Name of experimental data file.")
    parser.add_argument("--fixations-file-name", type=str,
                        default=os.path.join(os.path.dirname(
                            os.path.realpath(__file__)), "data/fixations.csv"),
                        help="Name of fixations file.")
    parser.add_argument("--use-cis-trials", default=False, action="store_true",
                        help="Use CIS trials in the analysis.")
    parser.add_argument("--use-trans-trials", default=False,
                        action="store_true", help="Use TRANS trials in the "
                        "analysis.")
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
        args.expdata_file_name, args.fixations_file_name,
        convertItemValues=convert_item_values)

    # Begin maximum likelihood estimation using odd trials only.
    # Get correct subset of trials.
    dataTrials = list()
    subjectIds = args.subject_ids if args.subject_ids else data.keys()
    for subjectId in subjectIds:
        numTrials = (args.trials_per_subject if args.trials_per_subject >= 1
                     else len(data[subjectId]))
        isCisTrial = [True if trial.valueLeft * trial.valueRight >= 0
                      else False for trial in data[subjectId]]
        isTransTrial = [True if trial.valueLeft * trial.valueRight <= 0
                        else False for trial in data[subjectId]]
        if args.use_cis_trials and args.use_trans_trials:
            trialSet = np.random.choice(
                [trialId for trialId in range(len(data[subjectId]))
                 if trialId % 2],
                numTrials, replace=False)
        elif args.use_cis_trials and not args.use_trans_trials:
            trialSet = np.random.choice(
                [trialId for trialId in range(len(data[subjectId]))
                 if trialId % 2 and isCisTrial[trialId]],
                numTrials, replace=False)
        elif not args.use_cis_trials and args.use_trans_trials:
            trialSet = np.random.choice(
                [trialId for trialId in range(len(data[subjectId]))
                 if trialId % 2 and isTransTrial[trialId]],
                numTrials, replace=False)
        else:
            return
        dataTrials.extend([data[subjectId][t] for t in trialSet])

    # Create all models to be used in the grid search.
    models = list()
    for d in args.range_d:
        for sigma in args.range_sigma:
            for theta in args.range_theta:
                models.append(aDDM(d, sigma, theta))

    # Get likelihoods for all models.
    if args.verbose:
        print("Starting grid search...")
    likelihoods = dict()
    for model in models:
        if args.verbose:
            print("Computing likelihoods for model " + str(model.params) +
                  "...")
        try:
            likelihoods[model.params] = model.parallel_get_likelihoods(
                dataTrials, numThreads=args.num_threads)
        except:
            print("An exception occurred during the likelihood " +
                  "computations for model " + str(model.params) + ".")
            raise

    # Get negative log likelihoods and optimal parameters.
    NLL = dict()
    for model in models:
        NLL[model.params] = - np.sum(np.log(likelihoods[model.params]))
    optimalParams = min(NLL, key=NLL.get)

    if args.verbose:
        print("Finished grid search!")
        print("Optimal d: " + str(optimalParams[0]))
        print("Optimal sigma: " + str(optimalParams[1]))
        print("Optimal theta: " + str(optimalParams[2]))
        print("Min NLL: " + str(min(NLL.values())))

    # Get fixation distributions from even trials.
    if args.verbose:
        print("Getting fixation distributions from even trials...")
    fixationData = get_empirical_distributions(
        data, subjectIds=subjectIds, useOddTrials=False, useEvenTrials=True,
        useCisTrials=args.use_cis_trials, useTransTrials=args.use_trans_trials)

    # Generate simulations using the even trials fixation distributions and the
    # estimated parameters.
    model = aDDM(*optimalParams)
    simulTrials = list()
    orientations = range(-15,20,5)
    for orLeft in orientations:
        for orRight in orientations:
            if (orLeft == orRight or
                (not args.use_cis_trials and orLeft * orRight > 0) or
                (not args.use_trans_trials and orLeft * orRight < 0)):
                continue
            valueLeft = np.absolute((np.absolute(orLeft) - 15) / 5)
            valueRight = np.absolute((np.absolute(orRight) - 15) / 5)
            for s in range(args.simulations_per_condition):
                try:
                    simulTrials.append(
                        model.simulate_trial(valueLeft, valueRight,
                                             fixationData))
                except:
                    print("An exception occurred while generating " +
                          "artificial trial " + str(s) + " for condition (" +
                          str(valueLeft) + ", " + str(valueRight) + ").")
                    raise

    currTime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    if args.save_simulations:
        save_simulations_to_csv(simulTrials,
                                "simul_expdata_" + currTime + ".csv",
                                "simul_fixations_" + currTime + ".csv")

    if args.save_figures:
        pdfPages = PdfPages("addm_fit_" + currTime + ".pdf")
        generate_choice_curves(dataTrials, simulTrials, pdfPages)
        generate_rt_curves(dataTrials, simulTrials, pdfPages)
        pdfPages.close()


if __name__ == "__main__":
    main()
