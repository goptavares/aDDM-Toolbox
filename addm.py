#!/usr/bin/python

"""
addm.py
Author: Gabriela Tavares, gtavares@caltech.edu

Implementation of the attentional drift-diffusion model (aDDM), as described by
Krajbich et al. (2010).
"""

import matplotlib
matplotlib.use('Agg')

from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool
from scipy.stats import norm

import matplotlib.pyplot as plt
import numpy as np


class FixationData:
    def __init__(self, probFixLeftFirst, latencies, transitions, fixations,
                 fixDistType):
        """
        Args:
          probFixLeftFirst: float between 0 and 1, empirical probability that
              the left item will be fixated first.
          latencies: numpy array corresponding to the empirical distribution of
              trial latencies (delay before first fixation) in miliseconds.
          transitions: numpy array corresponding to the empirical distribution
              of transitions (delays between item fixations) in miliseconds.
          fixations: dict whose indexing is defined according to parameter
              fixDistType. Each entry is a numpy array corresponding to the
              empirical distribution of item fixation durations in miliseconds.
          fixDistType: string, one of {'simple', 'difficulty', 'fixation'},
              determines how the fixation distributions are indexed. If
              'simple', fixation distributions are indexed only by type (1st,
              2nd, etc). If 'difficulty', they are indexed by type and by trial
              difficulty. If 'fixation', they are indexed by type and by the
              value difference between the fixated and unfixated items.
        """
        if (fixDistType is not 'simple' and fixDistType is not 'difficulty' and
            fixDistType is not 'fixation'):
            raise RuntimeError("Argument fixDistType must be one of {'simple', "
                               "'difficulty', 'fixation'}")
        self.probFixLeftFirst = probFixLeftFirst
        self.latencies = latencies
        self.transitions = transitions
        self.fixations = fixations
        self.fixDistType = fixDistType


class aDDMTrial:
    def __init__(self, RT, choice, valueLeft, valueRight, fixItem=np.empty((0)),
                 fixTime=np.empty((0)), fixRDV=np.empty((0)),
                 uninterruptedLastFixTime=None, isCisTrial=False,
                 isTransTrial=False):
        """
        Args:
          RT: reaction time in miliseconds.
          choice: either -1 (for left item) or +1 (for right item).
          valueLeft: value of the left item.
          valueRight: value of the right item.
          fixItem: list of items fixated during the trial in chronological
              order; 1 correponds to left, 2 corresponds to right, and any other
              value is considered a transition/blank fixation.
          fixTime: list of fixation durations (in miliseconds) in chronological
              order.
          fixRDV: list of floats corresponding to the RDV values at the end of
            each fixation in the trial.
          uninterruptedLastFixTime: integer corresponding to the duration, in
            miliseconds, that the last fixation in the trial would have if it
            had not been interrupted when a decision was made.
        """
        self.RT = RT
        self.choice = choice
        self.valueLeft = valueLeft
        self.valueRight = valueRight
        self.fixItem = fixItem
        self.fixTime = fixTime
        self.fixRDV = fixRDV
        self.uninterruptedLastFixTime = uninterruptedLastFixTime
        self.isCisTrial = isCisTrial
        self.isTransTrial = isTransTrial


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


class aDDM:
    """
    Implementation of the attentional drift-diffusion model (aDDM), as described
    by Krajbich et al. (2010).
    """
    def __init__(self, d, sigma, theta, barrier=1):
        """
        Args:
          d: float, parameter of the model which controls the speed of
              integration of the signal.
          sigma: float, parameter of the model, standard deviation for the
              normal distribution.
          theta: float between 0 and 1, parameter of the model which controls
              the attentional bias.
          barrier: positive number, magnitude of the signal thresholds.
        """
        self.d = d
        self.sigma = sigma
        self.theta = theta
        self.barrier = barrier
        self.params = (d, sigma, theta)


    def get_trial_likelihood(self, trial, timeStep=10,  approxStateStep=0.1,
                             visualDelay=0, motorDelay=0, plotTrial=False):
        """
        Computes the likelihood of the data from a single trial for these
        particular aDDM parameters.
        Args:
          trial: aDDMTrial object.
          timeStep: integer, value in miliseconds to be used for binning the
              time axis.
          approxStateStep: float, to be used for binning the RDV axis.
          visualDelay: delay to be discounted from the beginning of all
              fixations, in miliseconds.
          motorDelay: delay to be discounted from the last fixation only, in
              miliseconds.
          plotTrial: boolean, flag that determines whether the algorithm
              evolution for the trial should be plotted.
        Returns:
          The likelihood obtained for the given trial and model.
        """
        # Iterate over the fixations and discount visual delay.
        if visualDelay > 0:
            correctedFixItem = list()
            correctedFixTime = list()
            for i in xrange(len(trial.fixItem)):
                if trial.fixItem[i] == 1 or trial.fixItem[i] == 2:
                    correctedFixItem.append(0)
                    correctedFixTime.append(min(visualDelay, trial.fixTime[i]))
                    correctedFixItem.append(trial.fixItem[i])
                    correctedFixTime.append(
                        max(trial.fixTime[i] - visualDelay, 0))
                else:
                    correctedFixItem.append(trial.fixItem[i])
                    correctedFixTime.append(trial.fixTime[i])
        else:
            correctedFixItem = list(trial.fixItem)
            correctedFixTime = list(trial.fixTime)

        # Iterate over the fixations and discount motor delay from last
        # fixation.
        if motorDelay > 0:
            for i in xrange(len(correctedFixItem) - 1, -1, -1):
                if correctedFixItem[i] == 1 or correctedFixItem[i] == 2:
                    correctedFixTime[i] = max(
                        correctedFixTime[i] - motorDelay, 0)
                    break

        # Iterate over the fixations and get the number of time steps for this
        # trial.
        numTimeSteps = 0
        for fTime in correctedFixTime:
            numTimeSteps += int(fTime // timeStep)
        if numTimeSteps < 1:
            raise RuntimeError("Trial reaction time is smaller than time step.")
        numTimeSteps += 1

        # The values of the barriers can change over time.
        decay = 0  # decay = 0 means barriers are constant.
        barrierUp = self.barrier * np.ones(numTimeSteps)
        barrierDown = -self.barrier * np.ones(numTimeSteps)
        for t in xrange(1, numTimeSteps):
            barrierUp[t] = float(self.barrier) / float(1 + (decay * t))
            barrierDown[t] = float(-self.barrier) / float(1 + (decay * t))

        # Obtain correct state step.
        halfNumStateBins = np.ceil(self.barrier / float(approxStateStep))
        stateStep = self.barrier / float(halfNumStateBins + 0.5)

        # The vertical axis is divided into states.
        states = np.arange(barrierDown[0] + (stateStep / 2.),
                           barrierUp[0] - (stateStep / 2.) + stateStep,
                           stateStep)

        # Initial probability for all states is zero, except the zero state, for
        # which the initial probability is one.
        prStates = np.zeros((states.size, numTimeSteps))
        prStates[np.where(states==0)[0],0] = 1

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
                mean = self.d * (-trial.valueRight +
                                 (self.theta * trial.valueLeft))
            else:
                mean = 0

            # Iterate over the time interval of this fixation.
            for t in xrange(int(fTime // timeStep)):
                # Update the probability of the states that remain inside the
                # barriers. The probability of being in state B is the sum, over
                # all states A, of the probability of being in A at the previous
                # timestep times the probability of changing from A to B. We
                # multiply the probability by the stateStep to ensure that the
                # area under the curves for the probability distributions
                # probUpCrossing and probDownCrossing add up to 1.
                prStatesNew = (
                    stateStep *
                    np.dot(norm.pdf(changeMatrix, mean, self.sigma),
                           prStates[:,time-1]))
                prStatesNew[(states >= barrierUp[time]) |
                            (states <= barrierDown[time])] = 0

                # Calculate the probabilities of crossing the up barrier and the
                # down barrier. This is given by the sum, over all states A, of
                # the probability of being in A at the previous timestep times
                # the probability of crossing the barrier if A is the previous
                # state.
                tempUpCross = np.dot(
                    prStates[:,time-1],
                    (1 - norm.cdf(changeUp[:, time], mean, self.sigma)))
                tempDownCross = np.dot(
                    prStates[:,time-1],
                    norm.cdf(changeDown[:, time], mean, self.sigma))

                # Renormalize to cope with numerical approximations.
                sumIn = np.sum(prStates[:,time-1])
                sumCurrent = np.sum(prStatesNew) + tempUpCross + tempDownCross
                prStatesNew = (prStatesNew * float(sumIn)) / float(sumCurrent)
                tempUpCross = (tempUpCross * float(sumIn)) / float(sumCurrent)
                tempDownCross = ((tempDownCross * float(sumIn)) /
                                 float(sumCurrent))

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
            currTime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            pp = PdfPages("addm_trial_" + currTime + ".pdf")
            title = ("value left = %d, value right = %d" %
                     (trial.valueLeft, trial.valueRight))

            # Choose a suitable normalization constant.
            maxProb = max(prStates[:,3])

            fig = plt.figure()
            plt.imshow(prStates[::-1,:], extent=[1, numTimeSteps,
                                                 -self.barrier,
                                                 self.barrier],
                       aspect="auto", vmin=0, vmax=maxProb)
            plt.title(title)
            pp.savefig(fig)
            plt.close(fig)

            fig = plt.figure()
            plt.plot(range(1, numTimeSteps + 1), probUpCrossing, label="up",
                     color="red")
            plt.plot(range(1, numTimeSteps + 1), probDownCrossing,
                     label="down", color="green")
            plt.xlabel("Time")
            plt.ylabel("P(crossing)")
            plt.legend()
            plt.title(title)
            pp.savefig(fig)
            plt.close(fig)

            probInner = np.sum(prStates, 0)
            probUp = np.cumsum(probUpCrossing)
            probDown = np.cumsum(probDownCrossing)
            probTotal = probInner + probUp + probDown
            fig = plt.figure()
            plt.plot(range(1, numTimeSteps + 1), probUp, color="red",
                     label="up")
            plt.plot(range(1, numTimeSteps + 1), probDown, color="green",
                     label="down")
            plt.plot(range(1, numTimeSteps + 1), probInner, color="yellow",
                     label="in")
            plt.plot(range(1, numTimeSteps + 1), probTotal, color="blue",
                     label="total")
            plt.axis([1, numTimeSteps, 0, 1.1])
            plt.xlabel("Time")
            plt.ylabel("Cumulative probability")
            plt.legend()
            plt.title(title)
            pp.savefig(fig)
            plt.close(fig)

            fig = plt.figure()
            plt.plot(range(1, numTimeSteps + 1), probTotal - 1)
            plt.xlabel("Time")
            plt.ylabel("Numerical error")
            plt.title(title)
            pp.savefig(fig)
            plt.close(fig)
            
            pp.close()

        return likelihood


    def parallel_get_likelihoods(self, trials=None, timeStep=10, stateStep=0.1,
                                 visualDelay=0, motorDelay=0, numThreads=4):
        """
        Uses a threadpool to computes the likelihood of the data from a set of
        aDDM trials for these particular aDDM parameters.
        Args:
          addmTrials: list of aDDMTrial objects.
          timeStep: integer, value in miliseconds to be used for binning the
              time axis.
          stateStep: float, to be used for binning the RDV axis.
          visualDelay: delay to be discounted from the beginning of all
              fixations, in miliseconds.
          motorDelay: delay to be discounted from the last fixation only, in
              miliseconds.
          numThreads: int, number of threads to be used in the threadpool.
        Returns:
          A list of likelihoods obtained for the given trials and model.
        """
        pool = Pool(numThreads)
        likelihoods = pool.map(unwrap_addm_get_trial_likelihood,
                               zip([self] * len(trials),
                                   trials,
                                   [timeStep] * len(trials),
                                   [stateStep] * len(trials),
                                   [visualDelay] * len(trials),
                                   [motorDelay] * len(trials)))
        pool.close()
        return likelihoods


    def simulate_trial(self, valueLeft, valueRight, fixationData, timeStep=10,
                       numFixDists=3, visualDelay=0, motorDelay=0,
                       fixationDist=None, timeBins=None):
        """
        Generates an aDDM trial given the item values and some empirical
        fixation data, which are used to generate the simulated fixations.
        Args:
          valueLeft: value of the left item.
          valueRight: value of the right item.
          fixationData: a FixationData object.
          timeStep: integer, value in miliseconds to be used for binning the
              time axis.
          numFixDists: integer, number of fixation types to use in the fixation
              distributions. For instance, if numFixDists equals 3, then 3
              separate fixation types will be used, corresponding to the 1st,
              2nd and other (3rd and up) fixations in each trial.
          visualDelay: delay to be discounted from the beginning of all
              fixations, in miliseconds.
          motorDelay: delay to be discounted from the last fixation only, in
              miliseconds.
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
        RDV = 0
        trialTime = 0

        # Sample and iterate over the latency for this trial.
        trialAborted = False
        while True:
            latency = np.random.choice(fixationData.latencies)
            for t in xrange(int(latency // timeStep)):
                # Sample the change in RDV from the distribution.
                RDV += np.random.normal(0, self.sigma)
                # If the RDV hit one of the barriers, we abort the trial,
                # since a trial must end on an item fixation.
                if RDV >= self.barrier or RDV <= -self.barrier:
                    trialAborted = True
                    break

            if trialAborted:
                RDV = 0
                trialAborted = False
                continue
            else:
                # Add latency to this trial's data.
                fixRDV.append(RDV)
                fixItem.append(0)
                fixTime.append(latency - (latency % timeStep))
                trialTime += latency - (latency % timeStep)
                break

        # Sample the first fixation for this trial.
        probLeftRight = np.array([fixationData.probFixLeftFirst,
                                  1 - fixationData.probFixLeftFirst])
        currFixItem = np.random.choice([1, 2], p=probLeftRight)
        if not fixationDist:
            if fixationData.fixDistType == 'simple':
                currFixTime = np.random.choice(
                    fixationData.fixations[1]) - visualDelay
            elif fixationData.fixDistType == 'difficulty':
                valueDiff = np.absolute(valueLeft - valueRight)
                currFixTime = np.random.choice(
                    fixationData.fixations[1][valueDiff]) - visualDelay
            elif fixationData.fixDistType == 'fixation':
                valueDiff = fixUnfixValueDiffs[currFixItem]
                currFixTime = np.random.choice(
                    fixationData.fixations[1][valueDiff]) - visualDelay
        else:
            valueDiff = fixUnfixValueDiffs[currFixItem]
            prob = ([value for (key, value) in
                     sorted(fixationDist[1][valueDiff].items())])
            currFixTime = np.random.choice(timeBins, p=prob) - visualDelay

        # Iterate over all fixations in this trial.
        fixNumber = 2
        trialFinished = False
        while True:
            # Iterate over the visual delay for the current fixation.
            for t in xrange(int(visualDelay // timeStep)):
                # Sample the change in RDV from the distribution.
                RDV += np.random.normal(0, sigma)

                # If the RDV hit one of the barriers, the trial is over.
                if RDV >= self.barrier or RDV <= -self.barrier:
                    if RDV >= self.barrier:
                        choice = -1
                    elif RDV <= -self.barrier:
                        choice = 1
                    fixRDV.append(RDV)
                    fixItem.append(currFixItem)
                    fixTime.append(((t + 1) * timeStep) + motorDelay)
                    trialTime += ((t + 1) * timeStep) + motorDelay
                    RT = trialTime
                    uninterruptedLastFixTime = currFixTime
                    trialFinished = True
                    break

            if trialFinished:
                break

            # Iterate over the time interval of the current fixation.
            for t in xrange(int((currFixTime - visualDelay) // timeStep)):
                # We use a distribution to model changes in RDV
                # stochastically. The mean of the distribution (the change
                # most likely to occur) is calculated from the model
                # parameters and from the values of the two items.
                if currFixItem == 1:  # Subject is looking left.
                    mean = self.d * (valueLeft - (self.theta * valueRight))
                elif currFixItem == 2:  # Subject is looking right.
                    mean = self.d * (-valueRight + (self.theta * valueLeft))

                # Sample the change in RDV from the distribution.
                RDV += np.random.normal(mean, self.sigma)

                # If the RDV hit one of the barriers, the trial is over.
                if RDV >= self.barrier or RDV <= -self.barrier:
                    if RDV >= self.barrier:
                        choice = -1
                    elif RDV <= -self.barrier:
                        choice = 1
                    fixRDV.append(RDV)
                    fixItem.append(currFixItem)
                    fixTime.append(
                        ((t + 1) * timeStep) + visualDelay + motorDelay)
                    trialTime += ((t + 1) * timeStep) + visualDelay + motorDelay
                    RT = trialTime
                    uninterruptedLastFixTime = currFixTime
                    trialFinished = True
                    break

            if trialFinished:
                break

            # Add previous fixation to this trial's data.
            fixRDV.append(RDV)
            fixItem.append(currFixItem)
            fixTime.append(currFixTime - (currFixTime % timeStep))
            trialTime += currFixTime - (currFixTime % timeStep)

            # Sample and iterate over transition time.
            transitionTime = np.random.choice(fixationData.transitions)
            for t in xrange(int(transitionTime // timeStep)):
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
                    fixTime.append(((t + 1) * timeStep) + motorDelay)
                    trialTime += (((t + 1) * timeStep) + motorDelay)
                    RT = trialTime
                    uninterruptedLastFixTime = currFixTime
                    trialFinished = True
                    break

            if trialFinished:
                break

            # Add previous transition to this trial's data.
            fixRDV.append(RDV)
            fixItem.append(0)
            fixTime.append(transitionTime - (transitionTime % timeStep))
            trialTime += transitionTime - (transitionTime % timeStep)

            # Sample the next fixation for this trial.
            if currFixItem == 1:
                currFixItem = 2
            elif currFixItem == 2:
                currFixItem = 1
            if not fixationDist:
                if fixationData.fixDistType == 'simple':
                    currFixTime = np.random.choice(
                        fixationData.fixations[fixNumber]) - visualDelay
                elif fixationData.fixDistType == 'difficulty':
                    valueDiff = np.absolute(valueLeft - valueRight)
                    currFixTime = (np.random.choice(
                        fixationData.fixations[fixNumber][valueDiff]) -
                        visualDelay)
                elif fixationData.fixDistType == 'fixation':
                    valueDiff = fixUnfixValueDiffs[currFixItem]
                    currFixTime = (np.random.choice(
                        fixationData.fixations[fixNumber][valueDiff]) -
                        visualDelay)
            else:
                valueDiff = fixUnfixValueDiffs[currFixItem]
                prob = ([value for (key, value) in
                         sorted(fixationDist[fixNumber][valueDiff].items())])
                currFixTime = np.random.choice(timeBins, p=prob) - visualDelay

            if fixNumber < numFixDists:
                fixNumber += 1

        return aDDMTrial(RT, choice, valueLeft, valueRight, fixItem, fixTime,
                         fixRDV, uninterruptedLastFixTime)
