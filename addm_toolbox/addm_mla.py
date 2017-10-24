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

Module: addm_mla.py
Author: Gabriela Tavares, gtavares@caltech.edu

Maximum likelihood algorithm for the attentional drift-diffusion model (aDDM).
This algorithm uses response time histograms conditioned on choice from both
data and simulations to estimate each model's log-likelihood.
"""

from __future__ import division, absolute_import

import numpy as np

from builtins import range, str

from .addm import aDDMTrial
from .ddm_mla import DDM


class aDDM(DDM):
    """
    Implementation of the attentional drift-diffusion model (aDDM), as
    described by Krajbich et al. (2010).
    """
    def __init__(self, d, sigma, theta, barrier=1, nonDecisionTime=0, bias=0):
        """
        Args:
          d: float, parameter of the model which controls the speed of
              integration of the signal.
          sigma: float, parameter of the model, standard deviation for the
              normal distribution.
          theta: float between 0 and 1, parameter of the model which controls
              the attentional bias.
          barrier: positive number, magnitude of the signal thresholds.
          nonDecisionTime: non-negative integer, the amount of time in
              milliseconds during which only noise is added to the decision
              variable.
          bias: number, corresponds to the initial value of the decision
              variable. Must be smaller than barrier.
        """
        DDM.__init__(self, d, sigma, barrier, nonDecisionTime, bias)
        self.theta = theta
        self.params = (d, sigma, theta)


    def simulate_trial(self, valueLeft, valueRight, fixationData, timeStep=10,
                       numFixDists=3):
        """
        Generates an aDDM trial given the item values and some empirical
        fixation data, which are used to generate the simulated fixations.
        Args:
          valueLeft: value of the left item.
          valueRight: value of the right item.
          fixationData: a FixationData object.
          timeStep: integer, value in milliseconds to be used for binning the
              time axis.
          numFixDists: integer, number of fixation types to use in the fixation
              distributions. For instance, if numFixDists equals 3, then 3
              separate fixation types will be used, corresponding to the 1st,
              2nd and other (3rd and up) fixations in each trial.
        Returns:
          An aDDMTrial object resulting from the simulation.
        """
        RDV = self.bias
        RT = 0
        trialTime = 0
        choice = 0
        fixItem = list()
        fixTime = list()
        fixRDV = list()

        # Sample and iterate over the latency for this trial.
        latency = np.random.choice(fixationData.latencies)
        remainingNDT = self.nonDecisionTime - latency
        for t in range(int(latency // timeStep)):
            # Sample the change in RDV from the distribution.
            RDV += np.random.normal(0, self.sigma)

            # If the RDV hit one of the barriers, the trial is over.
            if RDV >= self.barrier or RDV <= -self.barrier:
                if RDV >= self.barrier:
                    choice = -1
                elif RDV <= -self.barrier:
                    choice = 1
                fixRDV.append(RDV)
                fixItem.append(0)
                fixTime.append((t + 1) * timeStep)
                trialTime += ((t + 1) * timeStep)
                RT = trialTime
                return aDDMTrial(RT, choice, valueLeft, valueRight, fixItem,
                                 fixTime, fixRDV)

        # Add latency to this trial's data.
        fixRDV.append(RDV)
        fixItem.append(0)
        fixTime.append(latency - (latency % timeStep))
        trialTime += latency - (latency % timeStep)

        fixUnfixValueDiffs = {1: valueLeft - valueRight,
                              2: valueRight - valueLeft}

        fixNumber = 1
        prevFixatedItem = -1
        currFixLocation = 0
        decisionReached = False

        while True:
            if currFixLocation == 0:
                # This is an item fixation; sample its location.
                if prevFixatedItem == -1:
                    # Sample the first item fixation for this trial.
                    probLeftRight = np.array(
                        [fixationData.probFixLeftFirst,
                         1 - fixationData.probFixLeftFirst])
                    currFixLocation = np.random.choice([1, 2], p=probLeftRight)
                elif prevFixatedItem == 1:
                    currFixLocation = 2
                elif prevFixatedItem == 2:
                    currFixLocation = 1
                prevFixatedItem = currFixLocation
                # Sample the duration of this item fixation.
                valueDiff = fixUnfixValueDiffs[currFixLocation]
                currFixTime = np.random.choice(
                    fixationData.fixations[fixNumber][valueDiff])
                if fixNumber < numFixDists:
                    fixNumber += 1
            else:
                # This is a transition.
                currFixLocation = 0
                # Sample the duration of this transition.
                currFixTime = np.random.choice(fixationData.transitions)

            # Iterate over the remaining non-decision time.
            if remainingNDT > 0:
                for t in range(int(remainingNDT // timeStep)):
                    # Sample the change in RDV from the distribution.
                    RDV += np.random.normal(0, self.sigma)

                    # If the RDV hit one of the barriers, the trial is over.
                    if RDV >= self.barrier or RDV <= -self.barrier:
                        if RDV >= self.barrier:
                            choice = -1
                        elif RDV <= -self.barrier:
                            choice = 1
                        fixRDV.append(RDV)
                        fixItem.append(currFixLocation)
                        fixTime.append((t + 1) * timeStep)
                        trialTime += ((t + 1) * timeStep)
                        RT = trialTime
                        uninterruptedLastFixTime = currFixTime
                        decisionReached = True
                        break

            if decisionReached:
                break

            remainingFixTime = max(0, currFixTime - max(0, remainingNDT))
            remainingNDT -= currFixTime

            # Iterate over the duration of the current fixation.
            for t in range(int(remainingFixTime // timeStep)):
                epsilon = np.random.normal(0, self.sigma)
                if currFixLocation == 0:
                    RDV += epsilon
                elif currFixLocation == 1:
                    RDV += (self.d * (
                        valueLeft - (self.theta * valueRight))) + epsilon
                elif currFixLocation == 2:
                    RDV += (self.d * (
                        (self.theta * valueLeft) - valueRight)) + epsilon

                if RDV >= self.barrier or RDV <= -self.barrier:
                    if RDV >= self.barrier:
                        choice = -1
                    elif RDV <= -self.barrier:
                        choice = 1
                    fixRDV.append(RDV)
                    fixItem.append(currFixLocation)
                    fixTime.append((t + 1) * timeStep)
                    trialTime += ((t + 1) * timeStep)
                    RT = trialTime
                    decisionReached = True
                    break

            if decisionReached:
                break

            # Add fixation to this trial's data.
            fixRDV.append(RDV)
            fixItem.append(currFixLocation)
            fixTime.append(currFixTime - (currFixTime % timeStep))
            trialTime += currFixTime - (currFixTime % timeStep)

        return aDDMTrial(RT, choice, valueLeft, valueRight, fixItem, fixTime,
                         fixRDV)


    def get_model_log_likelihood(self, fixationData, trialConditions,
                             numSimulations, histBins, dataHistLeft,
                             dataHistRight):
        """
        Computes the log-likelihood of a data set given the parameters of the
        aDDM. Data set is provided in the form of response time histograms
        conditioned on choice.
        Args:
          fixationData: a FixationData object.
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
              The log-likelihood for the given data and model.
        """
        logLikelihood = 0
        for trialCondition in trialConditions:
            RTsLeft = list()
            RTsRight = list()
            sim = 0
            while sim < numSimulations:
                try:
                    addmTrial = self.simulate_trial(
                        trialCondition[0], trialCondition[1], fixationData)
                except:
                    print(u"An exception occurred while generating "
                          "artificial trial " + str(sim) + u" for condition " +
                          str(trialCondition) + u", during the "
                          "log-likelihood computation for model " +
                          str(self.params) + u".")
                    raise
                if addmTrial.choice == -1:
                    RTsLeft.append(addmTrial.RT)
                elif addmTrial.choice == 1:
                    RTsRight.append(addmTrial.RT)
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
