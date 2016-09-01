#!/usr/bin/python

"""
simulate_addm_true_distributions.py
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

from multiprocessing import Pool

import argparse
import collections
import numpy as np
import pandas as pd
import sys

from addm import aDDM
from util import (load_data_from_csv, get_empirical_distributions,
                  save_simulations_to_csv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-ids", nargs="+", type=str, default=[],
                        help="List of subject ids. If not provided, all "
                        "existing subjects will be used.")
    parser.add_argument("--num-threads", type=int, default=9,
                        help="Size of the thread pool.")
    parser.add_argument("--bin-step", type=int, default=10,
                        help="Size of the bin step to be used in the fixation "
                        "distributions.")
    parser.add_argument("--max-fix-bin", type=int, default=3000,
                        help="Maximum fixation length to be used in the "
                        "fixation distributions.")
    parser.add_argument("--num-fix-dists", type=int, default=3,
                        help="Number of fixation distributions.")
    parser.add_argument("--num-iterations", type=int, default=3,
                        help="Number of iterations used to approximate the"
                        "true distributions.")
    parser.add_argument("--num-simulations", type=int, default=400,
                        help="Number of simulations to be generated per trial "
                        "condition.")
    parser.add_argument("--d", type=float, default=0.004,
                        help="aDDM parameter for generating simulations.")
    parser.add_argument("--sigma", type=float, default=0.07,
                        help="aDDM parameter for generating simulations.")
    parser.add_argument("--theta", type=float, default=0.25,
                        help="aDDM parameter for generating simulations.")
    parser.add_argument("--expdata-file-name", type=str, default="expdata.csv",
                        help="Name of experimental data file.")
    parser.add_argument("--fixations-file-name", type=str,
                        default="fixations.csv",
                        help="Name of fixations file.")
    parser.add_argument("--save-simulations", default=False,
                        action="store_true", help="Save simulations to CSV.")
    parser.add_argument("--verbose", default=False, action="store_true",
                        help="Increase output verbosity.")
    args = parser.parse_args()

    # Time bins to be used in the fixation distributions.
    bins = range(args.bin_step, args.max_fix_bin + args.bin_step,
                 args.bin_step)

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

    # Get fixation distributions from even trials.
    subjectIds = args.subject_ids if args.subject_ids else data.keys()
    try:
        fixationData = get_empirical_distributions(
            data, subjectIds=subjectIds, useOddTrials=False,
            useEvenTrials=True)
    except:
        print("An exception occurred while getting fixation distributions.")
        raise

    # Trial conditions for generating simulations.
    orientations = range(-15,20,5)
    trialConditions = list()
    for oLeft in orientations:
        for oRight in orientations:
            if oLeft != oRight:
                vLeft = np.absolute((np.absolute(oLeft) - 15) / 5)
                vRight = np.absolute((np.absolute(oRight) - 15) / 5)
                trialConditions.append((vLeft, vRight))

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
        for trialCondition in trialConditions:
            for s in range(args.num_simulations):
                try:
                    simulTrials.append(model.simulate_trial(
                        trialCondition[0], trialCondition[1], fixationData,
                        numFixDists=args.num_fix_dists,
                        fixationDist=empiricalFixDist, timeBins=bins))
                except Exception as e:
                    print("An exception occurred while running simulations "
                          "for trial condition " + str(trialCondition) + 
                          ", iteration " + str(it) + ": " + str(e))
                    return

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
            lastFixItem = -1
            numFix = 1
            for item, time in zip(trial.fixItem[::-1], trial.fixTime[::-1]):
                if not lastItemFixSkipped and (item == 1 or item == 2):
                    lastFixItem = item
                    lastItemFixSkipped = True
                    continue
                if item == 1 or item == 2:
                    bin = args.bin_step * min((time // args.bin_step) + 1,
                                              len(bins))
                    vDiff = fixUnfixValueDiffs[item]
                    countTotal[numFix][vDiff][bin] += 1
                    if numFix < args.num_fix_dists:
                        numFix += 1
            # Count last fixation.
            vDiff = fixUnfixValueDiffs[lastFixItem]
            bin = args.bin_step * min(
                (trial.uninterruptedLastFixTime // args.bin_step) + 1,
                len(bins))
            countLastFix[numFix][vDiff][bin] += 1
            countTotal[numFix][vDiff][bin] += 1

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
    for trialCondition in trialConditions:
        for s in range(args.num_simulations):
            try:
                simulTrials.append(model.simulate_trial(
                    trialCondition[0], trialCondition[1], fixationData,
                    numFixDists=args.num_fix_dists,
                    fixationDist=empiricalFixDist, timeBins=bins))
            except Exception as e:
                print("An exception occurred while running final simulations "
                      "for trial condition " + str(trialCondition) + ": " +
                      str(e))
                return

    if args.save_simulations:
        save_simulations_to_csv(simulTrials)


if __name__ == '__main__':
    main()
