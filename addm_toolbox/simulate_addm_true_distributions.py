#!/usr/bin/env python

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

from __future__ import absolute_import, division

import numpy as np
import pkg_resources

from builtins import range, str, zip
from datetime import datetime

from .addm import aDDM
from .util import (load_trial_conditions_from_csv, load_data_from_csv,
                   get_empirical_distributions, save_simulations_to_csv,
                   convert_item_values)



def main(d, sigma, theta, trialsFileName=None, expdataFileName=None,
         fixationsFileName=None, binStep=10, maxFixBin=3000, numFixDists=3,
         numIterations=3, simulationsPerCondition=800, subjectIds=[],
         saveSimulations=False, verbose=False):
    """
    Args:
      d: float, aDDM parameter for generating artificial data.
      sigma: float, aDDM parameter for generating artificial data.
      theta: float, aDDM parameter for generating artificial data.
      trialsFileName: string, path of trial conditions file.
      expdataFileName: string, path of experimental data file.
      fixationsFileName: string, path of fixations file.
      binStep: int, size of the bin step to be used in the fixation
          distributions.
      maxFixBin: int, maximum fixation length to be used in the fixation
          distributions.
      numFixDists: int, number of fixation distributions.
      numIterations: int, number of iterations used to approximate the true
          distributions.
      simulationsPerCondition: int, number of simulations to be generated per
          trial condition.
      subjectIds: list of strings corresponding to the subject ids. If not
          provided, all existing subjects will be used.
      saveFigures: boolean, whether or not save figures comparing choice and RT
          curves for data and simulations.
      verbose: boolean, whether or not to increase output verbosity.
    """
    # Load trial conditions.
    if not trialsFileName:
        trialsFileName = pkg_resources.resource_filename(
            u"addm_toolbox", u"data/trial_conditions.csv")
    trialConditions = load_trial_conditions_from_csv(trialsFileName)

    # Load experimental data from CSV file.
    if verbose:
        print(u"Loading experimental data...")
    if not expdataFileName:
        expdataFileName = pkg_resources.resource_filename(
            u"addm_toolbox", u"data/expdata.csv")
    if not fixationsFileName:
        fixationsFileName = pkg_resources.resource_filename(
            u"addm_toolbox", u"data/fixations.csv")
    data = load_data_from_csv(expdataFileName, fixationsFileName,
                              convertItemValues=convert_item_values)

    # Time bins to be used in the fixation distributions.
    bins = list(range(binStep, maxFixBin + binStep, binStep))

    # Get fixation distributions from even trials.
    if verbose:
        print(u"Getting fixation distributions from even trials...")
    subjectIds = ([str(subj) for subj in subjectIds] if subjectIds
                  else list(data))
    fixationData = get_empirical_distributions(
        data, subjectIds=subjectIds, useOddTrials=False, useEvenTrials=True)

    # Create original empirical distributions of fixations.
    empiricalFixDist = dict()
    for numFix in range(1, numFixDists + 1):
        empiricalFixDist[numFix] = dict()
        for valueDiff in range(-3,4):
            empiricalFixDist[numFix][valueDiff] = dict()
            for bin in bins:
                empiricalFixDist[numFix][valueDiff][bin] = 0
            for fixTime in fixationData.fixations[numFix][valueDiff]:
                bin = binStep * min((fixTime // binStep) + 1, len(bins))
                empiricalFixDist[numFix][valueDiff][bin] += 1

    # Normalize the distributions.
    for numFix in range(1, numFixDists + 1):
        for valueDiff in range(-3,4):
            sumBins = sum(list(empiricalFixDist[numFix][valueDiff].values()))
            for bin in bins:
                empiricalFixDist[numFix][valueDiff][bin] = (
                    empiricalFixDist[numFix][valueDiff][bin] / sumBins)

    model = aDDM(d, sigma, theta)
    for it in range(numIterations):
        if verbose:
            print(u"Iteration " + str(it + 1) + u"/" + str(numIterations))
        # Generate simulations using the current empirical distributions and
        # the model parameters.
        simulTrials = list()
        for (valueLeft, valueRight) in trialConditions:
            for s in range(simulationsPerCondition):
                try:
                    simulTrials.append(
                        model.simulate_trial(
                            valueLeft, valueRight, fixationData,
                            numFixDists=numFixDists,
                            fixationDist=empiricalFixDist, timeBins=bins))
                except:
                    print(u"An exception occurred while generating "
                          "artificial trial " + str(s) + u" for condition (" +
                          str(valueLeft) + u", " + str(valueRight) +
                          u") (iteration " + str(it) + u").")
                    raise

        countLastFix = dict()
        countTotal = dict()
        for numFix in range(1, numFixDists + 1):
            countLastFix[numFix] = dict()
            countTotal[numFix] = dict()
            for valueDiff in range(-3,4):
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
            for item, time in zip(reversed(trial.fixItem),
                                  reversed(trial.fixTime)):
                if not lastItemFixSkipped and (item == 1 or item == 2):
                    # Count last fixation (only if it was to an item).
                    bin = binStep * min(
                        (trial.uninterruptedLastFixTime // binStep) + 1,
                        len(bins))
                    vDiff = fixUnfixValueDiffs[item]
                    countLastFix[numFix][vDiff][bin] += 1
                    countTotal[numFix][vDiff][bin] += 1
                    lastItemFixSkipped = True
                    continue
                if item == 1 or item == 2:
                    # Count item fixations other than the last one.
                    bin = binStep * min((time // binStep) + 1, len(bins))
                    vDiff = fixUnfixValueDiffs[item]
                    countTotal[numFix][vDiff][bin] += 1
                    if numFix < numFixDists:
                        numFix += 1

        # Obtain true distributions of fixations.
        trueFixDist = dict()
        for numFix in range(1, numFixDists + 1):
            trueFixDist[numFix] = dict()
            for valueDiff in range(-3,4):
                trueFixDist[numFix][valueDiff] = dict()
                for bin in bins:
                    probNotLastFix = 1
                    if countTotal[numFix][valueDiff][bin] > 0:
                        probNotLastFix = 1 - (
                            countLastFix[numFix][valueDiff][bin] /
                            countTotal[numFix][valueDiff][bin])
                    if probNotLastFix == 0:
                        trueFixDist[numFix][valueDiff][bin] = (
                            empiricalFixDist[numFix][valueDiff][bin])
                    else:
                        trueFixDist[numFix][valueDiff][bin] = (
                            empiricalFixDist[numFix][valueDiff][bin] /
                            probNotLastFix)
        # Normalize the distributions.
        for numFix in range(1, numFixDists + 1):
            for valueDiff in range(-3,4):
                sumBins = sum(list(trueFixDist[numFix][valueDiff].values()))
                if sumBins > 0:
                    for bin in bins:
                        trueFixDist[numFix][valueDiff][bin] = (
                            trueFixDist[numFix][valueDiff][bin] /
                            sumBins)

        # Update empirical distributions using the current true distributions.
        empiricalFixDist = trueFixDist

    # Generate final simulations.
    simulTrials = list()
    for (valueLeft, valueRight) in trialConditions:
        for s in range(simulationsPerCondition):
            try:
                simulTrials.append(
                    model.simulate_trial(
                        valueLeft, valueRight, fixationData,
                        numFixDists=numFixDists,
                        fixationDist=empiricalFixDist, timeBins=bins))
            except:
                print(u"An exception occurred while generating "
                      "artificial trial " + str(s) + u" for condition (" +
                      str(valueLeft) + u", " + str(valueRight) +
                      u") in the final simulations generation.")
                raise

    if saveSimulations:
        currTime = datetime.now().strftime(u"%Y-%m-%d_%H:%M:%S")
        save_simulations_to_csv(simulTrials,
                                u"simul_expdata_" + currTime + u".csv",
                                u"simul_fixations_" + currTime + u".csv")
