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

Module: ddm_mla_test.py
Author: Gabriela Tavares, gtavares@caltech.edu

Performs a test to check the validity of the maximum likelihood algorithm (MLA)
for the drift-diffusion model (DDM). Artificial data is generated using
specific parameters for the model. These parameters are then recovered through
a maximum likelihood estimation procedure, using a grid search over the 2 free
parameters of the model.
"""

from __future__ import absolute_import

import numpy as np
import pkg_resources

from builtins import range, str
from multiprocessing import Pool

from .ddm_mla import DDM
from .util import load_trial_conditions_from_csv


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


def main(d, sigma, rangeD, rangeSigma, trialsFileName=None, numTrials=10,
         numSimulations=10, binStep=100, maxRT=8000, numThreads=9,
         verbose=False):
    """
    Args:
      d: float, DDM parameter for generating artificial data.
      sigma: float, DDM parameter for generating artificial data.
      rangeD: list of floats, search range for parameter d.
      rangeSigma: list of floats, search range for parameter sigma.
      trialsFileName: string, path of trial conditions file.
      numTrials: int, number of artificial data trials to be generated per
          trial condition.
      numSimulations: int, number of simulations to be generated per trial
          condition, to be used in the RT histograms.
      binStep: int, size of the bin step to be used in the RT histograms.
      maxRT: int, maximum RT to be used in the RT histograms.
      numThreads: int, size of the thread pool.
      verbose: boolean, whether or not to increase output verbosity.
    """
    pool = Pool(numThreads)

    histBins = list(range(0, maxRT + binStep, binStep))

    # Load trial conditions.
    if not trialsFileName:
        trialsFileName = pkg_resources.resource_filename(
            u"addm_toolbox", u"test_data/test_trial_conditions.csv")
    trialConditions = load_trial_conditions_from_csv(trialsFileName)

    # Generate artificial data.
    dataRTLeft = dict()
    dataRTRight = dict()
    for trialCondition in trialConditions:
        dataRTLeft[trialCondition] = list()
        dataRTRight[trialCondition] = list()
    model = DDM(d, sigma)
    for trialCondition in trialConditions:
        t = 0
        while t < numTrials:
            try:
                trial = model.simulate_trial(
                    trialCondition[0], trialCondition[1])
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
    if verbose:
        print(u"Performing grid search over the model parameters...")
    listParams = list()
    models = list()
    for d in rangeD:
        for sigma in rangeSigma:
            model = DDM(d, sigma)
            models.append(model)
            listParams.append((model, trialConditions, numSimulations,
                              histBins, dataHistLeft, dataHistRight))
    logLikelihoods = pool.map(wrap_ddm_get_model_log_likelihood, listParams)
    pool.close()

    if verbose:
        for i, model in enumerate(models):
            print(u"L" + str(model.params) + u" = " + str(logLikelihoods[i]))
        bestIndex = logLikelihoods.index(max(logLikelihoods))
        print(u"Best fit: " + str(models[bestIndex].params))
