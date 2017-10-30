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

Module: addm_pta_test.py
Author: Gabriela Tavares, gtavares@caltech.edu

Test to check the validity of the aDDM parameter estimation. Artificial data is
generated using specific parameters for the model. Fixations are sampled from
the data pooled from all subjects (or from a subset of subjects, when
provided). The parameters used for data generation are then recovered through a
maximum a posteriori estimation procedure.
"""

from __future__ import absolute_import, division

import numpy as np
import pkg_resources

from builtins import range, str

from .addm import aDDM
from .util import (load_trial_conditions_from_csv, load_data_from_csv,
                   get_empirical_distributions, convert_item_values)


def main(d, sigma, theta, rangeD, rangeSigma, rangeTheta, trialsFileName=None,
         expdataFileName=None, fixationsFileName=None, trialsPerCondition=800,
         subjectIds=[], numThreads=9, verbose=False):
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
      trialsPerCondition: int, number of artificial data trials to be generated
          per trial condition.
      subjectIds: list of strings corresponding to the subject ids. If not
          provided, all existing subjects will be used.
      numThreads: int, size of the thread pool.
      verbose: boolean, whether or not to increase output verbosity.
    """
    # Load trial conditions.
    if not trialsFileName:
        trialsFileName = pkg_resources.resource_filename(
            u"addm_toolbox", u"test_data/test_trial_conditions.csv")
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

    # Get fixation distributions.
    if verbose:
        print(u"Getting fixation distributions...")
    fixationData = get_empirical_distributions(data, subjectIds=subjectIds)

    # Generate artificial data.
    if verbose:
        print(u"Generating artificial data...")
    model = aDDM(d, sigma, theta)
    trials = list()
    for (valueLeft, valueRight) in trialConditions:
        for t in range(trialsPerCondition):
            try:
                trials.append(
                    model.simulate_trial(valueLeft, valueRight, fixationData))
            except:
                print(u"An exception occurred while generating artificial "
                      "trial " + str(t) + u" for condition (" +
                      str(valueLeft) + u", " + str(valueRight) + u").")
                raise

    # Get likelihoods for all models and all artificial trials.
    numModels = (len(rangeD) * len(rangeSigma) * len(rangeTheta))
    likelihoods = dict()
    models = list()
    posteriors = dict()
    for d in rangeD:
        for sigma in rangeSigma:
            for theta in rangeTheta:
                model = aDDM(d, sigma, theta)
                if verbose:
                    print(u"Computing likelihoods for model " +
                          str(model.params) + u"...")
                try:
                    likelihoods[model.params] = model.parallel_get_likelihoods(
                        trials, numThreads=numThreads)
                except:
                    print(u"An exception occurred during the likelihood "
                          "computations for model " + str(model.params) + u".")
                    raise
                models.append(model)
                posteriors[model.params] = 1 / numModels

    # Compute the posteriors.
    for t in range(len(trials)):
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

    if verbose:
        for model in models:
            print(u"P" + str(model.params) +  u" = " +
                  str(posteriors[model.params]))
        print(u"Sum: " + str(sum(list(posteriors.values()))))
