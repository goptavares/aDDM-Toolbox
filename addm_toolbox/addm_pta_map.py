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

Module: addm_pta_map.py
Author: Gabriela Tavares, gtavares@caltech.edu

Posterior distribution estimation procedure for the attentional drift-diffusion
model (aDDM), using a grid search over the 3 free parameters of the model. Data
from all subjects is pooled such that a single set of optimal parameters is
estimated (or from a subset of subjects, when provided).

aDDM simulations are generated according to the posterior distribution obtained
(instead of generating simulations from a single model, we sample models from
the posterior distribution and simulate them, then aggregate all simulations).
"""

from __future__ import absolute_import, division

import numpy as np
import pkg_resources

from builtins import range, str
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages

from .addm import aDDM
from .util import (load_trial_conditions_from_csv, load_data_from_csv,
                   get_empirical_distributions, save_simulations_to_csv,
                   generate_choice_curves, generate_rt_curves,
                   convert_item_values)


def main(rangeD, rangeSigma, rangeTheta, trialsFileName=None,
         expdataFileName=None, fixationsFileName=None, trialsPerSubject=100,
         numSamples=100, numSimulations=800, subjectIds=[], numThreads=9,
         saveSimulations=False, saveFigures=False, verbose=False):
    """
    Args:
      rangeD: list of floats, search range for parameter d.
      rangeSigma: list of floats, search range for parameter sigma.
      rangeTheta: list of floats, search range for parameter theta.
      trialsFileName: string, path of trial conditions file.
      expdataFileName: string, path of experimental data file.
      fixationsFileName: string, path of fixations file.
      trialsPerSubject: int, number of trials from each subject to be used in
          the analysis. If smaller than 1, all trials are used.
      numSamples: int, number of samples to be drawn from the posterior
          distribution when generating simulations.
      numSimulations: int, number of simulations to be genearated for each
          sample drawn from the posterior distribution and for each trial
          condition.
      subjectIds: list of strings corresponding to the subject ids. If not
          provided, all existing subjects will be used.
      numThreads: int, size of the thread pool.
      saveSimulations: boolean, whether or not to save simulations to CSV.
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

    # Begin posterior estimation using odd trials only.
    # Get correct subset of trials.
    dataTrials = list()
    subjectIds = ([str(subj) for subj in subjectIds] if subjectIds
                  else list(data))
    for subjectId in subjectIds:
        maxNumTrials = len(data[subjectId]) // 2
        numTrials = (trialsPerSubject
                     if 1 <= trialsPerSubject <= maxNumTrials
                     else maxNumTrials)
        trialSet = np.random.choice(
            [trialId for trialId in range(len(data[subjectId]))
             if trialId % 2], numTrials, replace=False)
        dataTrials.extend([data[subjectId][t] for t in trialSet])

    # Create all models to be used in the grid search.
    numModels = (len(rangeD) * len(rangeTheta) * len(rangeSigma))
    models = list()
    posteriors = dict()
    for d in rangeD:
        for sigma in rangeSigma:
            for theta in rangeTheta:
                model = aDDM(d, sigma, theta)
                models.append(model)
                posteriors[model.params] = 1 / numModels

    # Get likelihoods for all models.
    if verbose:
        print(u"Starting grid search...")
    likelihoods = dict()
    for model in models:
        if verbose:
            print(u"Computing likelihoods for model " + str(model.params) +
                  u"...")
        try:
            likelihoods[model.params] = model.parallel_get_likelihoods(
                dataTrials, numThreads=numThreads)
        except:
            print(u"An exception occurred during the likelihood "
                  "computations for model " + str(model.params) + u".")
            raise

    if verbose:
        print(u"Finished grid search!")

    # Compute posterior distribution over all models.
    for t in range(len(dataTrials)):
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
            posteriors[model.params] = (likelihoods[model.params][t] * prior /
                                        denominator)

    if verbose:
        for model in models:
            print(u"P" + str(model.params) + u" = " +
                  str(posteriors[model.params]))

    # Get fixation distributions from even trials.
    if verbose:
        print(u"Getting fixation distributions from even trials...")
    fixationData = get_empirical_distributions(
        data, subjectIds=subjectIds, useOddTrials=False, useEvenTrials=True,
        maxFixTime=3000)

    # Get list of posterior distribution values.
    posteriorsList = list()
    for model in models:
        posteriorsList.append(posteriors[model.params])

    # Generate probabilistic set of simulations using the posterior
    # distribution.
    simulTrials = list()
    for s in range(numSamples):
        # Sample model from posteriors distribution.
        modelIndex = np.random.choice(
            np.array(list(range(numModels))), p=np.array(posteriorsList))
        model = models[modelIndex]
        for (valueLeft, valueRight) in trialConditions:
            for t in range(numSimulations):
                try:
                    simulTrials.append(
                        model.simulate_trial(valueLeft, valueRight,
                                             fixationData))
                except:
                    print(u"An exception occurred while generating "
                          "artificial trial " + str(t) + u" for "
                          "condition (" + str(valueLeft) + u", " +
                          str(valueRight) + u") and model " +
                          str(model.params) + u" (sample " + str(s) + u").")
                    raise

    currTime = datetime.now().strftime(u"%Y-%m-%d_%H:%M:%S")

    if saveSimulations:
        save_simulations_to_csv(simulTrials,
                                u"simul_expdata_" + currTime + u".csv",
                                u"simul_fixations_" + currTime + u".csv")

    if saveFigures:
        pdfPages = PdfPages(u"addm_fit_" + currTime + u".pdf")
        generate_choice_curves(dataTrials, simulTrials, pdfPages)
        generate_rt_curves(dataTrials, simulTrials, pdfPages)
        pdfPages.close()
