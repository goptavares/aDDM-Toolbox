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

Module: basinhopping_optimize.py
Author: Gabriela Tavares, gtavares@caltech.edu

Maximum likelihood estimation procedure for the attentional drift-diffusion
model (aDDM), using a Basinhopping algorithm to search the parameter space.
Data from all subjects is pooled such that a single set of optimal parameters
is estimated.
"""

from __future__ import absolute_import, division

import numpy as np
import pkg_resources

from builtins import range, str
from scipy.optimize import basinhopping

from .addm import aDDM
from .util import load_data_from_csv, convert_item_values


# Global variables.
dataTrials = []


def get_model_nll(params):
    """
    Computes the negative log likelihood of the global data set given the
    parameters of the aDDM.
    Args:
      params: list containing the 3 model parameters, in the following order:
          d, theta, sigma.
    Returns:
      The negative log likelihood for the global data set and the given model.
    """
    d = params[0]
    sigma = params[1]
    theta = params[2]
    model = aDDM(d, sigma, theta) 

    logLikelihood = 0
    for trial in dataTrials:
        try:
            likelihood = model.get_trial_likelihood(trial)
        except:
            print(u"An exception occurred during the likelihood " +
                  "computations for model " + str(model.params) + u".")
            raise
        if likelihood != 0:
            logLikelihood += np.log(likelihood)

    print(u"NLL for " + str(params) + u": " + str(-logLikelihood))
    if logLikelihood != 0:
        return -logLikelihood
    else:
        return float("inf")


def main(initialD, initialSigma, initialTheta, lowerBoundD=0.0001,
         upperBoundD=0.09, lowerBoundSigma=0.001, upperBoundSigma=0.9,
         lowerBoundTheta=0, upperBoundTheta=1, expdataFileName=None,
         fixationsFileName=None, trialsPerSubject=100, numIterations=100,
         stepSize=0.001, subjectIds=[], verbose=False):
    """
    Args:
      initialD: float, initial value for parameter d.
      initialSigma: float, initial value for parameter sigma.
      initialTheta: float, initial value for parameter theta.
      lowerBoundD: float, lower search bound for parameter d.
      upperBoundD: float, upper search bound for parameter d.
      lowerBoundSigma: float, lower search bound for parameter sigma.
      upperBoundSigma: float, upper search bound for parameter sigma.
      lowerBoundTheta: float, lower search bound for parameter theta.
      upperBoundTheta: float, upper search bound for parameter theta.
      expdataFileName: string, path of experimental data file.
      fixationsFileName: string, path of fixations file.
      trialsPerSubject: int, number of trials from each subject to be used in
          the analysis. If smaller than 1, all trials are used.
      numIterations: int, number of basin hopping iterations.
      stepSize: float, step size for use in the random displacement of the
          basin hopping algorithm.
      subjectIds: list of strings corresponding to the subject ids. If not
          provided, all existing subjects will be used.
      verbose: boolean, whether or not to increase output verbosity.
    """
    global dataTrials

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

    # Get correct subset of trials.
    subjectIds = ([str(subj) for subj in subjectIds] if subjectIds
                  else list(data))
    for subjectId in subjectIds:
        numTrials = (trialsPerSubject if trialsPerSubject >= 1
                     else len(data[subjectId]))
        trialSet = np.random.choice(
            [trialId for trialId in range(len(data[subjectId]))],
            numTrials, replace=False)
        dataTrials.extend([data[subjectId][t] for t in trialSet])

    # Initial guess for the parameters: d, sigma, theta.
    initialParams = [initialD, initialSigma, initialTheta]

    # Search bounds.
    bounds = [(lowerBoundD, upperBoundD),
              (lowerBoundSigma, upperBoundSigma),
              (lowerBoundTheta, upperBoundTheta)
             ]

    # Optimize using Basinhopping algorithm.
    minimizerKwargs = dict(method=u"L-BFGS-B", bounds=bounds)
    result = basinhopping(
        get_model_nll, initialParams, minimizer_kwargs=minimizerKwargs,
        niter=numIterations,stepsize=stepSize)
    print(u"Optimization result: " + str(result))
