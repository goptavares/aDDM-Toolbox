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

Module: addm_pta_mle.py
Author: Gabriela Tavares, gtavares@caltech.edu

Maximum likelihood estimation procedure for the attentional drift-diffusion
model (aDDM), using a grid search over the 3 free parameters of the model. Data
from all subjects is pooled such that a single set of optimal parameters is
estimated (or from a subset of subjects, when provided).

aDDM simulations are generated for the model with maximum estimated likelihood.
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
         simulationsPerCondition=800, subjectIds=[], numThreads=9,
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
      simulationsPerCondition: int, number of simulations to be generated per
          trial condition.
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

    # Begin maximum likelihood estimation using odd trials only.
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
    models = list()
    for d in rangeD:
        for sigma in rangeSigma:
            for theta in rangeTheta:
                models.append(aDDM(d, sigma, theta))

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

    # Get negative log likelihoods and optimal parameters.
    NLL = dict()
    for model in models:
        NLL[model.params] = - np.sum(np.log(likelihoods[model.params]))
    optimalParams = min(NLL, key=NLL.get)

    if verbose:
        print(u"Finished grid search!")
        print(u"Optimal d: " + str(optimalParams[0]))
        print(u"Optimal sigma: " + str(optimalParams[1]))
        print(u"Optimal theta: " + str(optimalParams[2]))
        print(u"Min NLL: " + str(min(list(NLL.values()))))

    # Get fixation distributions from even trials.
    if verbose:
        print(u"Getting fixation distributions from even trials...")
    fixationData = get_empirical_distributions(
        data, subjectIds=subjectIds, useOddTrials=False, useEvenTrials=True)

    # Generate simulations using the even trials fixation distributions and the
    # estimated parameters.
    if verbose:
        print(u"Generating model simulations...")
    model = aDDM(*optimalParams)
    simulTrials = list()
    for (valueLeft, valueRight) in trialConditions:
        for s in range(simulationsPerCondition):
            try:
                simulTrials.append(
                    model.simulate_trial(valueLeft, valueRight, fixationData))
            except:
                print(u"An exception occurred while generating artificial "
                      "trial " + str(s) + u" for condition (" +
                      str(valueLeft) + u", " + str(valueRight) + u").")
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
