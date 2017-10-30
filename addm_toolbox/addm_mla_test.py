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

Module: addm_mla_test.py
Author: Gabriela Tavares, gtavares@caltech.edu

Performs a test to check the validity of the maximum likelihood algorithm (MLA)
for the attentional drift-diffusion model (aDDM). Artificial data is generated
using specific parameters for the model. These parameters are then recovered
through a maximum likelihood estimation procedure, using a grid search over the
3 free parameters of the model.
"""

from __future__ import absolute_import

import numpy as np
import pkg_resources

from builtins import range, str
from multiprocessing import Pool

from .addm_mla import aDDM
from .util import (load_trial_conditions_from_csv, load_data_from_csv,
                   get_empirical_distributions, convert_item_values)


def wrap_addm_get_model_log_likelihood(args):
    """
    Wrapper for aDDM.get_model_log_likelihood(), intended for parallel
    computation using a threadpool.
    Args:
      args: a tuple where the first item is an aDDM object, and the remaining
          item are the same arguments required by
          aDDM.get_model_log_likelihood().
    Returns:
      The output of aDDM.get_model_log_likelihood().
    """
    model = args[0]
    return model.get_model_log_likelihood(*args[1:])


def main(d, sigma, theta, rangeD, rangeSigma, rangeTheta, trialsFileName=None,
         expdataFileName=None, fixationsFileName=None, numTrials=10,
         numSimulations=10, subjectIds=[], binStep=100, maxRT=8000,
         numThreads=9, verbose=False):
    """
    Args:
      d: float, aDDM parameter for generating artificial data.
      sigma: float, aDDM parameter for generating artificial data.
      theta: float, aDDM parameter for generating artificial data.
      rangeD: list of floats, search range for parameter d.
      rangeSigma: list of floats, search range for parameter sigma.
      rangeTheta: list of floats, search range for parameter theta.
      trialsFileName: string, path of trial conditions file.
      expdataFileName: string, path of experimental data file.
      fixationsFileName: string, path of fixations file.
      numTrials: int, number of artificial data trials to be generated per
          trial condition.
      numSimulations: int, number of simulations to be generated per trial
          condition, to be used in the RT histograms.
      subjectIds: list of strings corresponding to the subject ids. If not
          provided, all existing subjects will be used.
      binStep: int, size of the bin step to be used in the RT histograms.
      maxRT: int, maximum RT to be used in the RT histograms.
      numThreads: int, size of the thread pool.
      verbose: boolean, whether or not to increase output verbosity.
    """
    pool = Pool(numThreads)

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

    # Get fixation distributions.
    if verbose:
        print(u"Getting fixation distributions...")
    fixationData = get_empirical_distributions(data, subjectIds=subjectIds)

    histBins = list(range(0, maxRT + binStep, binStep))

    # Load trial conditions.
    if not trialsFileName:
        trialsFileName = pkg_resources.resource_filename(
            u"addm_toolbox", u"test_data/test_trial_conditions.csv")
    trialConditions = load_trial_conditions_from_csv(trialsFileName)

    # Generate histograms for artificial data.
    dataHistLeft = dict()
    dataHistRight = dict()
    model = aDDM(d, sigma, theta)
    for trialCondition in trialConditions:
        RTsLeft = list()
        RTsRight = list()
        t = 0
        while t < numTrials:
            try:
                trial = model.simulate_trial(
                    trialCondition[0], trialCondition[1], fixationData)
            except:
                print(u"An exception occurred while generating artificial "
                      "trial " + str(t) + u" for condition " +
                      str(trialCondition[0]) + u", " + str(trialCondition[1]) +
                      u".")
                raise
            if trial.choice == -1:
                RTsLeft.append(trial.RT)
            elif trial.choice == 1:
                RTsRight.append(trial.RT)
            t += 1
        dataHistLeft[trialCondition] = np.histogram(RTsLeft, bins=histBins)[0]
        dataHistRight[trialCondition] = np.histogram(
            RTsRight, bins=histBins)[0]

    if verbose:
        print(u"Done generating histograms of artificial data!")
    
    # Grid search on the parameters of the model.
    if verbose:
        print(u"Performing grid search over the model parameters...")
    listParams = list()
    models = list()
    for d in rangeD:
        for sigma in rangeSigma:
            for theta in rangeTheta:
                model = aDDM(d, sigma, theta)
                models.append(model)
                listParams.append((model, fixationData, trialConditions,
                                   numSimulations, histBins, dataHistLeft,
                                   dataHistRight))
    logLikelihoods = pool.map(wrap_addm_get_model_log_likelihood, listParams)
    pool.close()

    if verbose:
        for i, model in enumerate(models):
            print(u"L" + str(model.params) + u" = " + str(logLikelihoods[i]))
        bestIndex = logLikelihoods.index(max(logLikelihoods))
        print(u"Best fit: " + str(models[bestIndex].params))
