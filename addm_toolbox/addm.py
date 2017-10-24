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

Module: addm.py
Author: Gabriela Tavares, gtavares@caltech.edu

Implementation of the attentional drift-diffusion model (aDDM), as described by
Krajbich et al. (2010).
"""

from __future__ import division, absolute_import

import matplotlib.pyplot as plt
import numpy as np

from builtins import range, str, zip
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool
from scipy.stats import norm

from .ddm import DDMTrial, DDM


class FixationData:
    def __init__(self, probFixLeftFirst, latencies, transitions, fixations,
                 fixDistType):
        """
        Args:
          probFixLeftFirst: float between 0 and 1, empirical probability that
              the left item will be fixated first.
          latencies: numpy array corresponding to the empirical distribution of
              trial latencies (delay before first fixation) in milliseconds.
          transitions: numpy array corresponding to the empirical distribution
              of transitions (delays between item fixations) in milliseconds.
          fixations: dict whose indexing is defined according to parameter
              fixDistType. Each entry is a numpy array corresponding to the
              empirical distribution of item fixation durations in
              milliseconds.
          fixDistType: string, one of {'simple', 'difficulty', 'fixation'},
              determines how the fixation distributions are indexed. If
              'simple', fixation distributions are indexed only by type (1st,
              2nd, etc). If 'difficulty', they are indexed by type and by trial
              difficulty, i.e., the absolute value for the trial's value
              difference. If 'fixation', they are indexed by type and by the
              value difference between the fixated and unfixated items.
        """
        fixDistType = str(fixDistType)
        if (fixDistType != u"simple" and fixDistType != u"difficulty"
            and fixDistType != u"fixation"):
            raise RuntimeError(u"Argument fixDistType must be one of {simple, "
                               "difficulty, fixation}")
        self.probFixLeftFirst = probFixLeftFirst
        self.latencies = latencies
        self.transitions = transitions
        self.fixations = fixations
        self.fixDistType = fixDistType


class aDDMTrial(DDMTrial):
    def __init__(self, RT, choice, valueLeft, valueRight,
                 fixItem=np.empty((0)), fixTime=np.empty((0)),
                 fixRDV=np.empty((0)), uninterruptedLastFixTime=None):
        """
        Args:
          RT: response time in milliseconds.
          choice: either -1 (for left item) or +1 (for right item).
          valueLeft: value of the left item.
          valueRight: value of the right item.
          fixItem: list of items fixated during the trial in chronological
              order; 1 correponds to left, 2 corresponds to right, and any
              other value is considered a transition/blank fixation.
          fixTime: list of fixation durations (in milliseconds) in
              chronological order.
          fixRDV: list of floats corresponding to the RDV values at the end of
            each fixation in the trial.
          uninterruptedLastFixTime: integer corresponding to the duration, in
            milliseconds, that the last fixation in the trial would have if it
            had not been interrupted when a decision was made.
        """
        DDMTrial.__init__(self, RT, choice, valueLeft, valueRight)
        self.fixItem = fixItem
        self.fixTime = fixTime
        self.fixRDV = fixRDV
        self.uninterruptedLastFixTime = uninterruptedLastFixTime


def unwrap_addm_get_trial_likelihood(arg, **kwarg):
    """
    Wrapper for aDDM.get_trial_likelihood(), intended for parallel computation
    using a threadpool. This method should stay outside the class, allowing it
    to be pickled (as required by multiprocessing).
    Args:
      params: same arguments required by aDDM.get_trial_likelihood().
    Returns:
      The output of aDDM.get_trial_likelihood().
    """
    return aDDM.get_trial_likelihood(*arg, **kwarg)


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


    def get_trial_likelihood(self, trial, timeStep=10, approxStateStep=0.1,
                             plotTrial=False):
        """
        Computes the likelihood of the data from a single trial for these
        particular aDDM parameters.
        Args:
          trial: aDDMTrial object.
          timeStep: integer, value in milliseconds to be used for binning the
              time axis.
          approxStateStep: float, to be used for binning the RDV axis.
          plotTrial: boolean, flag that determines whether the algorithm
              evolution for the trial should be plotted.
        Returns:
          The likelihood obtained for the given trial and model.
        """
        # Iterate over the fixations and discount the non-decision time.
        if self.nonDecisionTime > 0:
            correctedFixItem = []
            correctedFixTime = []
            remainingNDT = self.nonDecisionTime
            for fItem, fTime in zip(trial.fixItem, trial.fixTime):
                if remainingNDT > 0:
                    correctedFixItem.append(0)
                    correctedFixTime.append(min(remainingNDT, fTime))
                    correctedFixItem.append(fItem)
                    correctedFixTime.append(max(fTime - remainingNDT, 0))
                else:
                    correctedFixItem.append(fItem)
                    correctedFixTime.append(fTime)
        else:
            correctedFixItem = list(trial.fixItem)
            correctedFixTime = list(trial.fixTime)

        # Iterate over the fixations and get the number of time steps for this
        # trial.
        numTimeSteps = 0
        for fTime in correctedFixTime:
            numTimeSteps += int(fTime // timeStep)
        if numTimeSteps < 1:
            raise RuntimeError(u"Trial response time is smaller than time "
                               "step.")
        numTimeSteps += 1

        # The values of the barriers can change over time.
        decay = 0  # decay = 0 means barriers are constant.
        barrierUp = self.barrier * np.ones(numTimeSteps)
        barrierDown = -self.barrier * np.ones(numTimeSteps)
        for t in range(1, numTimeSteps):
            barrierUp[t] = self.barrier / (1 + (decay * t))
            barrierDown[t] = -self.barrier / (1 + (decay * t))

        # Obtain correct state step.
        halfNumStateBins = np.ceil(self.barrier / approxStateStep)
        stateStep = self.barrier / (halfNumStateBins + 0.5)

        # The vertical axis is divided into states.
        states = np.arange(barrierDown[0] + (stateStep / 2.),
                           barrierUp[0] - (stateStep / 2.) + stateStep,
                           stateStep)

        # Find the state corresponding to the bias parameter.
        biasState = np.argmin(np.absolute(states - self.bias))

        # Initial probability for all states is zero, except the bias state,
        # for which the initial probability is one.
        prStates = np.zeros((states.size, numTimeSteps))
        prStates[biasState,0] = 1

        # The probability of crossing each barrier over the time of the trial.
        probUpCrossing = np.zeros(numTimeSteps)
        probDownCrossing = np.zeros(numTimeSteps)

        time = 1

        changeMatrix = np.subtract(states.reshape(states.size, 1), states)
        changeUp = np.subtract(barrierUp, states.reshape(states.size, 1))
        changeDown = np.subtract(barrierDown, states.reshape(states.size, 1))

        # Iterate over all fixations in this trial.
        for fItem, fTime in zip(correctedFixItem, correctedFixTime):
            # We use a normal distribution to model changes in RDV
            # stochastically. The mean of the distribution (the change most
            # likely to occur) is calculated from the model parameters and from
            # the item values.
            if fItem == 1:  # Subject is looking left.
                mean = self.d * (trial.valueLeft -
                                 (self.theta * trial.valueRight))
            elif fItem == 2:  # Subject is looking right.
                mean = self.d * ((self.theta * trial.valueLeft) -
                                 trial.valueRight)
            else:
                mean = 0

            # Iterate over the time interval of this fixation.
            for t in range(int(fTime // timeStep)):
                # Update the probability of the states that remain inside the
                # barriers. The probability of being in state B is the sum,
                # over all states A, of the probability of being in A at the
                # previous timestep times the probability of changing from A to
                # B. We multiply the probability by the stateStep to ensure
                # that the area under the curves for the probability
                # distributions probUpCrossing and probDownCrossing add up to
                # 1.
                prStatesNew = (
                    stateStep *
                    np.dot(norm.pdf(changeMatrix, mean, self.sigma),
                           prStates[:, time-1]))
                prStatesNew[(states >= barrierUp[time]) |
                            (states <= barrierDown[time])] = 0

                # Calculate the probabilities of crossing the up barrier and
                # the down barrier. This is given by the sum, over all states
                # A, of the probability of being in A at the previous timestep
                # times the probability of crossing the barrier if A is the
                # previous state.
                tempUpCross = np.dot(
                    prStates[:, time-1],
                    (1 - norm.cdf(changeUp[:, time], mean, self.sigma)))
                tempDownCross = np.dot(
                    prStates[:, time-1],
                    norm.cdf(changeDown[:, time], mean, self.sigma))

                # Renormalize to cope with numerical approximations.
                sumIn = np.sum(prStates[:, time-1])
                sumCurrent = np.sum(prStatesNew) + tempUpCross + tempDownCross
                prStatesNew = prStatesNew * sumIn / sumCurrent
                tempUpCross = tempUpCross * sumIn / sumCurrent
                tempDownCross = tempDownCross * sumIn / sumCurrent

                # Update the probabilities of each state and the probabilities
                # of crossing each barrier at this timestep.
                prStates[:, time] = prStatesNew
                probUpCrossing[time] = tempUpCross
                probDownCrossing[time] = tempDownCross

                time += 1

        # Compute the likelihood contribution of this trial based on the final
        # choice.
        likelihood = 0
        if trial.choice == -1:  # Choice was left.
            if probUpCrossing[-1] > 0:
                likelihood = probUpCrossing[-1]
        elif trial.choice == 1:  # Choice was right.
            if probDownCrossing[-1] > 0:
                likelihood = probDownCrossing[-1]

        if plotTrial:
            currTime = datetime.now().strftime(u"%Y-%m-%d_%H:%M:%S")
            fileName = u"addm_trial_" + currTime + u".pdf"
            self.plot_trial(trial.valueLeft, trial.valueRight, timeStep,
                            numTimeSteps, prStates, probUpCrossing,
                            probDownCrossing, fileName=fileName)

        return likelihood


    def parallel_get_likelihoods(self, trials=None, timeStep=10, stateStep=0.1,
                                 numThreads=4):
        """
        Uses a threadpool to computes the likelihood of the data from a set of
        aDDM trials for these particular aDDM parameters.
        Args:
          addmTrials: list of aDDMTrial objects.
          timeStep: integer, value in milliseconds to be used for binning the
              time axis.
          stateStep: float, to be used for binning the RDV axis.
          numThreads: int, number of threads to be used in the threadpool.
        Returns:
          A list of likelihoods obtained for the given trials and model.
        """
        pool = Pool(numThreads)
        likelihoods = pool.map(unwrap_addm_get_trial_likelihood,
                               zip([self] * len(trials),
                                   trials,
                                   [timeStep] * len(trials),
                                   [stateStep] * len(trials)))
        pool.close()
        return likelihoods


    def simulate_trial(self, valueLeft, valueRight, fixationData, timeStep=10,
                       numFixDists=3, fixationDist=None, timeBins=None):
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
          fixationDist: distribution of fixations which, when provided, will be
              used instead of fixationData.fixations. This should be a dict of
              dicts of dicts, corresponding to the probability distributions of
              fixation durations. Indexed first by fixation type (1st, 2nd,
              etc), then by the value difference between the fixated and the
              unfixated items, then by time bin. Each entry is a number between
              0 and 1 corresponding to the probability assigned to the
              particular time bin (i.e. given a particular fixation type and
              value difference, probabilities for all bins should add up to 1).
          timeBins: list containing the time bins used in fixationDist.
        Returns:
          An aDDMTrial object resulting from the simulation.
        """
        fixUnfixValueDiffs = {1: valueLeft - valueRight,
                              2: valueRight - valueLeft}
        fixItem = list()
        fixTime = list()
        fixRDV = list()
        RDV = self.bias
        trialTime = 0

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
                uninterruptedLastFixTime = latency
                return aDDMTrial(RT, choice, valueLeft, valueRight, fixItem,
                                 fixTime, fixRDV, uninterruptedLastFixTime)

        # Add latency to this trial's data.
        fixRDV.append(RDV)
        fixItem.append(0)
        fixTime.append(latency - (latency % timeStep))
        trialTime += latency - (latency % timeStep)

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
                if not fixationDist:
                    if fixationData.fixDistType == u"simple":
                        currFixTime = np.random.choice(
                            fixationData.fixations[fixNumber])
                    elif fixationData.fixDistType == u"difficulty":
                        valueDiff = np.absolute(valueLeft - valueRight)
                        currFixTime = np.random.choice(
                            fixationData.fixations[fixNumber][valueDiff])
                    elif fixationData.fixDistType == u"fixation":
                        valueDiff = fixUnfixValueDiffs[currFixLocation]
                        currFixTime = np.random.choice(
                            fixationData.fixations[fixNumber][valueDiff])
                else:
                    valueDiff = fixUnfixValueDiffs[currFixLocation]
                    currFixTime = np.random.choice(
                        timeBins, p=[value for (key, value) in 
                        sorted(list(
                            fixationDist[fixNumber][valueDiff].items()))])
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
                # We use a distribution to model changes in RDV
                # stochastically. The mean of the distribution (the change
                # most likely to occur) is calculated from the model
                # parameters and from the values of the two items.
                if currFixLocation == 0:  # Transition.
                    mean = 0
                elif currFixLocation == 1:  # Subject is looking left.
                    mean = self.d * (valueLeft - (self.theta * valueRight))
                elif currFixLocation == 2:  # Subject is looking right.
                    mean = self.d * ((self.theta * valueLeft) - valueRight)

                # Sample the change in RDV from the distribution.
                RDV += np.random.normal(mean, self.sigma)

                # If the RDV hit one of the barriers, the trial is over.
                if RDV >= self.barrier or RDV <= -self.barrier:
                    if RDV >= self.barrier:
                        choice = -1
                    elif RDV <= -self.barrier:
                        choice = 1
                    fixRDV.append(RDV)
                    fixItem.append(currFixLocation)
                    fixTime.append((t + 1) * timeStep)
                    trialTime += (t + 1) * timeStep
                    RT = trialTime
                    uninterruptedLastFixTime = currFixTime
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
                         fixRDV, uninterruptedLastFixTime)
