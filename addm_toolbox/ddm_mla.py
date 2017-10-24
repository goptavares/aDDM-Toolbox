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

Module: ddm_mla.py
Author: Gabriela Tavares, gtavares@caltech.edu

Maximum likelihood algorithm for the classic drift-diffusion model (DDM). This
algorithm uses response time histograms conditioned on choice from both data
and simulations to estimate each model's log-likelihood.
"""

from __future__ import division, absolute_import

import numpy as np

from builtins import str

from .ddm import DDMTrial


class DDM(object):
    """
    Implementation of the traditional drift-diffusion model (DDM), as described
    by Ratcliff et al. (1998).
    """
    def __init__(self, d, sigma, barrier=1, nonDecisionTime=0, bias=0):
        """
        Args:
          d: float, parameter of the model which controls the speed of
              integration of the signal.
          sigma: float, parameter of the model, standard deviation for the
              normal distribution.
          barrier: positive number, magnitude of the signal thresholds.
          nonDecisionTime: non-negative integer, the amount of time in
              milliseconds during which only noise is added to the decision
              variable.
          bias: number, corresponds to the initial value of the decision
              variable. Must be smaller than barrier.
        """
        if barrier <= 0:
            raise ValueError(u"Error: barrier parameter must larger than "
                             "zero.")
        if bias >= barrier:
            raise ValueError(u"Error: bias parameter must be smaller than "
                             "barrier parameter.")
        self.d = d
        self.sigma = sigma
        self.barrier = barrier
        self.nonDecisionTime = nonDecisionTime
        self.bias = bias
        self.params = (d, sigma)


    def simulate_trial(self, valueLeft, valueRight, timeStep=10): 
        """
        DDM algorithm. Given the parameters of the model and the trial
        conditions, returns the choice and response time as predicted by the
        model.
        Args:
          valueLeft: integer, value of the left item.
          valueRight: integer, value of the right item.
          timeStep: integer, value in milliseconds which determines how often
              the RDV signal is updated.
        Returns:
          A DDMTrial object resulting from the simulation.
        """
        RT = 0
        choice = 0
        RDV = self.bias
        elapsedNDT = 0

        while RDV < self.barrier and RDV > -self.barrier:
            RT = RT + timeStep
            epsilon = np.random.normal(0, self.sigma)
            if elapsedNDT < self.nonDecisionTime // timeStep:
                RDV += epsilon
                elapsedNDT += 1
            else:
                RDV += (self.d * (valueLeft - valueRight)) + epsilon

        if RDV >= self.barrier:
            choice = -1
        elif RDV <= -self.barrier:
            choice = 1
        return DDMTrial(RT, choice, valueLeft, valueRight)


    def get_model_log_likelihood(self, trialConditions, numSimulations,
                                 histBins, dataHistLeft, dataHistRight):
        """
        Computes the log-likelihood of a data set given the model. Data set is
        provided in the form of response time histograms conditioned on choice.
        Args:
          trialConditions: list of pairs corresponding to the different trial
              conditions. Each pair contains the values of left and right
              items.
          numSimulations: integer, number of simulations per trial condition to
              be generated when creating response time histograms.
          histBins: list of numbers corresponding to the time bins used to
              create the response time histograms.
          dataHistLeft: dict indexed by trial condition (where each trial
              condition is a pair (valueLeft, valueRight)). Each entry is a
              numpy array corresponding to the response time histogram
              conditioned on left choice for the data. It is assumed that this
              histogram was created using the same time bins as argument
              histBins.
          dataHistRight: same as dataHistLeft, except that the response time
              histograms are conditioned on right choice.
          Returns:
              The log-likelihood for the data given the model.
        """
        logLikelihood = 0
        for trialCondition in trialConditions:
            RTsLeft = list()
            RTsRight = list()
            sim = 0
            while sim < numSimulations:
                try:
                    ddmTrial = self.simulate_trial(trialCondition[0],
                                                   trialCondition[1])
                except:
                    print(u"An exception occurred while generating "
                          "artificial trial " + str(sim) + u" for condition " +
                          str(trialCondition[0]) + u", " +
                          str(trialCondition[1]) + u", during the " +
                          u"log-likelihood computation for model " +
                          str(self.params) + u".")
                    raise
                if ddmTrial.choice == -1:
                    RTsLeft.append(ddmTrial.RT)
                elif ddmTrial.choice == 1:
                    RTsRight.append(ddmTrial.RT)
                sim += 1

            simulLeft = np.histogram(RTsLeft, bins=histBins)[0]
            if np.sum(simulLeft) != 0:
                simulLeft = simulLeft / np.sum(simulLeft)
            with np.errstate(divide=u"ignore"):
                logSimulLeft = np.where(simulLeft > 0, np.log(simulLeft), 0)
            dataLeft = np.array(dataHistLeft[trialCondition])
            logLikelihood += np.dot(logSimulLeft, dataLeft)

            simulRight = np.histogram(RTsRight, bins=histBins)[0]
            if np.sum(simulRight) != 0:
                simulRight = simulRight / np.sum(simulRight)
            with np.errstate(divide=u"ignore"):
                logSimulRight = np.where(simulRight > 0, np.log(simulRight), 0)
            dataRight = np.array(dataHistRight[trialCondition])
            logLikelihood += np.dot(logSimulRight, dataRight)

        return logLikelihood
