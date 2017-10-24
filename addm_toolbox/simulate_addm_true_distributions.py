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

Module: simulate_addm_true_distributions.py
Author: Gabriela Tavares, gtavares@caltech.edu

Generates aDDM simulations with an approximation of the "true" fixation
distributions. When creating fixation distributions, we leave out the last
fixation from each trial, since these are interrupted when a decision is made
and therefore their duration should not be sampled. Since long fixations are
more likely to be interrupted, they end up not being included in the
distributions. This means that the distributions we use to sample fixations are
biased towards shorter fixations than the "true" distributions. Here we use the
uninterrupted duration of last fixations to approximate the "true"
distributions of fixations. We do this by dividing each bin in the empirical
fixation distributions by the probability of a fixation in that bin being the
last fixation in the trial. The "true" distributions estimated are then used to
generate aDDM simulations.
"""

import argparse
import numpy as np
import os

from datetime import datetime

from addm import aDDM
from util import (load_trial_conditions_from_csv, load_data_from_csv,
                  get_empirical_distributions, save_simulations_to_csv,
                  convert_item_values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-ids", nargs="+", type=str, default=[],
                        help="List of subject ids. If not provided, all "
                        "existing subjects will be used.")
    parser.add_argument("--bin-step", type=int, default=10,
                        help="Size of the bin step to be used in the fixation "
                        "distributions.")
    parser.add_argument("--max-fix-bin", type=int, default=3000,
                        help="Maximum fixation length to be used in the "
                        "fixation distributions.")
    parser.add_argument("--num-fix-dists", type=int, default=3,
                        help="Number of fixation distributions.")
    parser.add_argument("--num-iterations", type=int, default=3,
                        help="Number of iterations used to approximate the "
                        "true distributions.")
    parser.add_argument("--simulations-per-condition", type=int,
                        default=800, help="Number of artificial data trials "
                        "to be generated per trial condition.")
    parser.add_argument("--d", type=float, default=0.004,
                        help="aDDM parameter for generating simulations.")
    parser.add_argument("--sigma", type=float, default=0.07,
                        help="aDDM parameter for generating simulations.")
    parser.add_argument("--theta", type=float, default=0.25,
                        help="aDDM parameter for generating simulations.")
    parser.add_argument("--trials-file-name", type=str,
                        default=os.path.join(
                            os.path.dirname(os.path.realpath(__file__)),
                            "data/trial_conditions.csv"),
                        help="Name of trial conditions file.")
    parser.add_argument("--expdata-file-name", type=str, 
                        default=os.path.join(os.path.dirname(
                            os.path.realpath(__file__)), "data/expdata.csv"),
                        help="Name of experimental data file.")
    parser.add_argument("--fixations-file-name", type=str,
                        default=os.path.join(os.path.dirname(
                            os.path.realpath(__file__)), "data/fixations.csv"),
                        help="Name of fixations file.")
    parser.add_argument("--save-simulations", default=False,
                        action="store_true", help="Save simulations to CSV.")
    parser.add_argument("--verbose", default=False, action="store_true",
                        help="Increase output verbosity.")
    args = parser.parse_args()

    # Load trial conditions.
    trialConditions = load_trial_conditions_from_csv(args.trials_file_name)

    # Time bins to be used in the fixation distributions.
    bins = range(args.bin_step, args.max_fix_bin + args.bin_step,
                 args.bin_step)

    # Load experimental data from CSV file.
    if args.verbose:
        print("Loading experimental data...")
    data = load_data_from_csv(
        args.expdata_file_name, args.fixations_file_name,
        convertItemValues=convert_item_values)

    # Get fixation distributions from even trials.
    if args.verbose:
        print("Getting fixation distributions from even trials...")
    subjectIds = args.subject_ids if args.subject_ids else data.keys()
    fixationData = get_empirical_distributions(
        data, subjectIds=subjectIds, useOddTrials=False, useEvenTrials=True)

    # Create original empirical distributions of fixations.
    empiricalFixDist = dict()
    for numFix in xrange(1, args.num_fix_dists + 1):
        empiricalFixDist[numFix] = dict()
        for valueDiff in xrange(-3,4):
            empiricalFixDist[numFix][valueDiff] = dict()
            for bin in bins:
                empiricalFixDist[numFix][valueDiff][bin] = 0
            for fixTime in fixationData.fixations[numFix][valueDiff]:
                bin = args.bin_step * min((fixTime // args.bin_step) + 1,
                                          len(bins))
                empiricalFixDist[numFix][valueDiff][bin] += 1

    # Normalize the distributions.
    for numFix in xrange(1, args.num_fix_dists + 1):
        for valueDiff in xrange(-3,4):
            sumBins = sum(empiricalFixDist[numFix][valueDiff].values())
            for bin in bins:
                empiricalFixDist[numFix][valueDiff][bin] = (
                    float(empiricalFixDist[numFix][valueDiff][bin]) /
                    float(sumBins))

    model = aDDM(args.d, args.sigma, args.theta)
    for it in xrange(args.num_iterations):
        if args.verbose:
            print("Iteration " + str(it + 1) + "/" + str(args.num_iterations))
        # Generate simulations using the current empirical distributions and
        # the model parameters.
        simulTrials = list()
        for (valueLeft, valueRight) in trialConditions:
            for s in range(args.simulations_per_condition):
                try:
                    simulTrials.append(
                        model.simulate_trial(
                            valueLeft, valueRight, fixationData,
                            numFixDists=args.num_fix_dists,
                            fixationDist=empiricalFixDist, timeBins=bins))
                except:
                    print("An exception occurred while generating " +
                          "artificial trial " + str(s) + " for condition (" +
                          str(valueLeft) + ", " + str(valueRight) +
                          ") (iteration " + str(it) + ").")
                    raise

        countLastFix = dict()
        countTotal = dict()
        for numFix in xrange(1, args.num_fix_dists + 1):
            countLastFix[numFix] = dict()
            countTotal[numFix] = dict()
            for valueDiff in xrange(-3,4):
                countLastFix[numFix][valueDiff] = dict()
                countTotal[numFix][valueDiff] = dict()
                for bin in bins:
                    countLastFix[numFix][valueDiff][bin] = 0
                    countTotal[numFix][valueDiff][bin] = 0

        for trial in simulTrials:
            # Count all item fixations, except last.
            fixUnfixValueDiffs = {
                1: trial.valueLeft - trial.valueRight,
                2: trial.valueRight - trial.valueLeft}
            lastItemFixSkipped = False
            numFix = 1
            for item, time in zip(trial.fixItem[::-1], trial.fixTime[::-1]):
                if not lastItemFixSkipped and (item == 1 or item == 2):
                    # Count last fixation (only if it was to an item).
                    bin = args.bin_step * min(
                        (trial.uninterruptedLastFixTime // args.bin_step) + 1,
                        len(bins))
                    vDiff = fixUnfixValueDiffs[item]
                    countLastFix[numFix][vDiff][bin] += 1
                    countTotal[numFix][vDiff][bin] += 1
                    lastItemFixSkipped = True
                    continue
                if item == 1 or item == 2:
                    # Count item fixations other than the last one.
                    bin = args.bin_step * min(
                        (time // args.bin_step) + 1, len(bins))
                    vDiff = fixUnfixValueDiffs[item]
                    countTotal[numFix][vDiff][bin] += 1
                    if numFix < args.num_fix_dists:
                        numFix += 1

        # Obtain true distributions of fixations.
        trueFixDist = dict()
        for numFix in xrange(1, args.num_fix_dists + 1):
            trueFixDist[numFix] = dict()
            for valueDiff in xrange(-3,4):
                trueFixDist[numFix][valueDiff] = dict()
                for bin in bins:
                    probNotLastFix = 1
                    if countTotal[numFix][valueDiff][bin] > 0:
                        probNotLastFix = 1 - (
                            float(countLastFix[numFix][valueDiff][bin]) /
                            float(countTotal[numFix][valueDiff][bin]))
                    if probNotLastFix == 0:
                        trueFixDist[numFix][valueDiff][bin] = (
                            empiricalFixDist[numFix][valueDiff][bin])
                    else:
                        trueFixDist[numFix][valueDiff][bin] = (
                            float(empiricalFixDist[numFix][valueDiff][bin]) /
                            float(probNotLastFix))
        # Normalize the distributions.
        for numFix in xrange(1, args.num_fix_dists + 1):
            for valueDiff in xrange(-3,4):
                sumBins = sum(trueFixDist[numFix][valueDiff].values())
                if sumBins > 0:
                    for bin in bins:
                        trueFixDist[numFix][valueDiff][bin] = (
                            float(trueFixDist[numFix][valueDiff][bin]) /
                            float(sumBins))

        # Update empirical distributions using the current true distributions.
        empiricalFixDist = trueFixDist

    # Generate final simulations.
    simulTrials = list()
    for (valueLeft, valueRight) in trialConditions:
        for s in range(args.simulations_per_condition):
            try:
                simulTrials.append(
                    model.simulate_trial(
                        valueLeft, valueRight, fixationData,
                        numFixDists=args.num_fix_dists,
                        fixationDist=empiricalFixDist, timeBins=bins))
            except:
                print("An exception occurred while generating " +
                      "artificial trial " + str(s) + " for condition (" +
                      str(valueLeft) + ", " + str(valueRight) +
                      ") in the final simulations generation.")
                raise

    if args.save_simulations:
        currTime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        save_simulations_to_csv(simulTrials,
                                "simul_expdata_" + currTime + ".csv",
                                "simul_fixations_" + currTime + ".csv")


if __name__ == "__main__":
    main()
