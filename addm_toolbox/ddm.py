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

Module: ddm.py
Author: Gabriela Tavares, gtavares@caltech.edu

Implementation of the classic drift-diffusion model (DDM), as described by
Ratcliff et al. (1998).
"""

from __future__ import division, absolute_import

import matplotlib.pyplot as plt
import numpy as np

from builtins import range, str, zip
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool
from scipy.stats import norm


class DDMTrial(object):
    def __init__(self, RT, choice, valueLeft, valueRight):
        """
        Args:
          RT: response time in milliseconds.
          choice: either -1 (for left item) or +1 (for right item).
          valueLeft: value of the left item.
          valueRight: value of the right item.
        """
        self.RT = RT
        self.choice = choice
        self.valueLeft = valueLeft
        self.valueRight = valueRight


def unwrap_ddm_get_trial_likelihood(arg, **kwarg):
    """
    Wrapper for DDM.get_trial_likelihood(), intended for parallel computation
    using a threadpool. This method should stay outside the class, allowing it
    to be pickled (as required by multiprocessing).
    Args:
      params: same arguments required by DDM.get_trial_likelihood().
    Returns:
      The output of DDM.get_trial_likelihood().
    """
    return DDM.get_trial_likelihood(*arg, **kwarg)


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
            raise ValueError("Error: barrier parameter must larger than zero.")
        if bias >= barrier:
            raise ValueError("Error: bias parameter must be smaller than "
                             "barrier parameter.")
        self.d = d
        self.sigma = sigma
        self.barrier = barrier
        self.nonDecisionTime = nonDecisionTime
        self.bias = bias
        self.params = (d, sigma)


    def get_trial_likelihood(self, trial, timeStep=10, approxStateStep=0.1,
                             plotTrial=False):
        """
        Computes the likelihood of the data from a single DDM trial for these
        particular DDM parameters.
        Args:
          trial: DDMTrial object.
          timeStep: integer, value in milliseconds to be used for binning the
              time axis.
          approxStateStep: float, to be used for binning the RDV axis.
          plotTrial: boolean, flag that determines whether the algorithm
              evolution for the trial should be plotted.
        Returns:
          The likelihood obtained for the given trial and model.
        """
        # Get the number of time steps for this trial.
        numTimeSteps = trial.RT // timeStep
        if numTimeSteps < 1:
            raise RuntimeError(u"Trial response time is smaller than time "
                               "step.")

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
        states = np.arange(barrierDown[0] + (stateStep / 2),
                           barrierUp[0] - (stateStep / 2) + stateStep,
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

        changeMatrix = np.subtract(states.reshape(states.size, 1), states)
        changeUp = np.subtract(barrierUp, states.reshape(states.size, 1))
        changeDown = np.subtract(barrierDown, states.reshape(states.size, 1))

        elapsedNDT = 0

        # Iterate over the time of this trial.
        for time in range(1, numTimeSteps):
            # We use a normal distribution to model changes in RDV
            # stochastically. The mean of the distribution (the change most
            # likely to occur) is calculated from the model parameter d and
            # from the item values, except during non-decision time, in which
            # the mean is zero.
            if elapsedNDT < self.nonDecisionTime // timeStep:
                mean = 0
                elapsedNDT += 1
            else:
                mean = self.d * (trial.valueLeft - trial.valueRight)

            # Update the probability of the states that remain inside the
            # barriers. The probability of being in state B is the sum, over
            # all states A, of the probability of being in A at the previous
            # time step times the probability of changing from A to B. We
            # multiply the probability by the stateStep to ensure that the area
            # under the curves for the probability distributions probUpCrossing
            # and probDownCrossing add up to 1.
            prStatesNew = (stateStep *
                           np.dot(norm.pdf(changeMatrix, mean, self.sigma),
                           prStates[:,time-1]))
            prStatesNew[(states >= barrierUp[time]) |
                        (states <= barrierDown[time])] = 0

            # Calculate the probabilities of crossing the up barrier and the
            # down barrier. This is given by the sum, over all states A, of the
            # probability of being in A at the previous timestep times the
            # probability of crossing the barrier if A is the previous state.
            tempUpCross = np.dot(
                prStates[:,time-1],
                (1 - norm.cdf(changeUp[:, time], mean, self.sigma)))
            tempDownCross = np.dot(
                prStates[:,time-1],
                norm.cdf(changeDown[:, time], mean, self.sigma))

            # Renormalize to cope with numerical approximations.
            sumIn = np.sum(prStates[:,time-1])
            sumCurrent = np.sum(prStatesNew) + tempUpCross + tempDownCross
            prStatesNew = prStatesNew * sumIn / sumCurrent
            tempUpCross = tempUpCross * sumIn / sumCurrent
            tempDownCross = tempDownCross * sumIn / sumCurrent

            # Update the probabilities of each state and the probabilities of
            # crossing each barrier at this timestep.
            prStates[:, time] = prStatesNew
            probUpCrossing[time] = tempUpCross
            probDownCrossing[time] = tempDownCross

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
            fileName = u"ddm_trial_" + currTime + u".pdf"
            self.plot_trial(trial.valueLeft, trial.valueRight, timeStep,
                            numTimeSteps, prStates, probUpCrossing,
                            probDownCrossing, fileName=fileName)

        return likelihood


    def parallel_get_likelihoods(self, ddmTrials, timeStep=10, stateStep=0.1,
                                 numThreads=4):
        """
        Uses a threadpool to compute the likelihood of the data from a set of
        DDM trials given the DDM parameters.
        Args:
          ddmTrials: list of DDMTrial objects.
          timeStep: integer, value in milliseconds to be used for binning the
              time axis.
          stateStep: float, to be used for binning the RDV axis.
          numThreads: int, number of threads to be used in the threadpool.
        Returns:
          A list of likelihoods obtained for the given trials and model.
        """
        pool = Pool(numThreads)
        likelihoods = pool.map(unwrap_ddm_get_trial_likelihood,
                               zip([self] * len(ddmTrials),
                                   ddmTrials,
                                   [timeStep] * len(ddmTrials),
                                   [stateStep] * len(ddmTrials)))
        pool.close()
        return likelihoods


    def simulate_trial(self, valueLeft, valueRight, timeStep=10):
        """
        Generates a DDM trial given the item values.
        Args:
          valueLeft: value of the left item.
          valueRight: value of the right item.
          timeStep: integer, value in milliseconds to be used for binning the
              time axis.
        Returns:
          A DDMTrial object resulting from the simulation.
        """
        RDV = self.bias
        time = 0
        elapsedNDT = 0
        while True:
            # If the RDV hit one of the barriers, the trial is over.
            if RDV >= self.barrier or RDV <= -self.barrier:
                RT = time * timeStep
                if RDV >= self.barrier:
                    choice = -1
                elif RDV <= -self.barrier:
                    choice = 1
                break

            if elapsedNDT < self.nonDecisionTime // timeStep:
                mean = 0
                elapsedNDT += 1
            else:
                mean = self.d * (valueLeft - valueRight)

            # Sample the change in RDV from the distribution.
            RDV += np.random.normal(mean, self.sigma)

            time += 1

        return DDMTrial(RT, choice, valueLeft, valueRight)


    def plot_trial(self, valueLeft, valueRight, timeStep, numTimeSteps,
                   probStates, probUpCrossing, probDownCrossing,
                   fileName=None):
        """
        Plots the likelihood computation for a trial and saves the figures to a
        PDF file.
        Args:
          valueLeft: value of the left item.
          valueRight: value of the right item.
          timeStep: integer, value in milliseconds used in the computation for
              binning the time axis.
          numTimeSteps: integer, number of time steps in the trial.
          probStates: 2-dimensional numpy array with size S x T, where S is the
              number of states and T is the number of time steps. Each value
              corresponds to the probability of the signa being at a specific
              state at a specific time in the trial. 
          probUpCrossing: numpy array with size T, where T is the number of
              time steps. Each value corresponds to the probability of crossing
              the upper barrier over the time of the trial.
          probDownCrossing: numpy array with size T, where T is the number of
              time steps. Each value corresponds to the probability of crossing
              the lower barrier over the time of the trial.
          fileName: string, name of the PDF file to save.
        """
        if not fileName:
            currTime = datetime.now().strftime(u"%Y-%m-%d_%H:%M:%S")
            fileName = u"trial_" + currTime + u".pdf"
        pp = PdfPages(fileName)

        title = u"value left = %d, value right = %d" % (valueLeft, valueRight)

        # Choose a suitable normalization constant.
        maxProb = max(probStates[:,3])

        fig, ax = plt.subplots()
        plt.imshow(probStates[::-1,:],
                   extent=[1, numTimeSteps, -self.barrier, self.barrier],
                   aspect=u"auto", vmin=0, vmax=maxProb)
        ax.set_xticklabels([str(x * timeStep) for x in ax.get_xticks()])
        plt.title(title)
        pp.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots()
        plt.plot(list(range(1, numTimeSteps + 1)), probUpCrossing,
                 label=u"up", color=u"red")
        plt.plot(list(range(1, numTimeSteps + 1)), probDownCrossing,
                 label=u"down", color=u"green")
        ax.set_xticklabels([str(x * timeStep) for x in ax.get_xticks()])
        plt.xlabel(u"Time")
        plt.ylabel(u"P(crossing)")
        plt.legend()
        plt.title(title)
        pp.savefig(fig)
        plt.close(fig)

        probInner = np.sum(probStates, 0)
        probUp = np.cumsum(probUpCrossing)
        probDown = np.cumsum(probDownCrossing)
        probTotal = probInner + probUp + probDown
        fig, ax = plt.subplots()
        plt.plot(list(range(1, numTimeSteps + 1)), probUp, color=u"red",
                 label=u"up")
        plt.plot(list(range(1, numTimeSteps + 1)), probDown,
                 color=u"green", label=u"down")
        plt.plot(list(range(1, numTimeSteps + 1)), probInner,
                 color=u"yellow", label=u"in")
        plt.plot(list(range(1, numTimeSteps + 1)), probTotal,
                 color=u"blue", label=u"total")
        plt.axis([1, numTimeSteps, 0, 1.1])
        ax.set_xticklabels([str(x * timeStep) for x in ax.get_xticks()])
        plt.xlabel(u"Time")
        plt.ylabel(u"Cumulative probability")
        plt.legend()
        plt.title(title)
        pp.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots()
        plt.plot(list(range(1, numTimeSteps + 1)), probTotal - 1)
        ax.set_xticklabels([str(x * timeStep) for x in ax.get_xticks()])
        plt.xlabel(u"Time")
        plt.ylabel(u"Numerical error")
        plt.title(title)
        pp.savefig(fig)
        plt.close(fig)
        
        pp.close()
