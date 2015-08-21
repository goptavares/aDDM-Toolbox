#!/usr/bin/python

"""
addm.py
Author: Gabriela Tavares, gtavares@caltech.edu

Implementation of the attentional drift-diffusion model (aDDM), as described by
Krajbich et al. (2010).
"""

import matplotlib
matplotlib.use('Agg')

from scipy.stats import norm

import collections
import matplotlib.pyplot as plt
import numpy as np


def get_trial_likelihood(choice, valueLeft, valueRight, fixItem, fixTime, d,
                         theta, sigma=0, mu=0, timeStep=10, stateStep=0.1,
                         barrier=1, visualDelay=0, motorDelay=0,
                         plotResults=False):
    """
    Computes the likelihood of the data from a single trial given a set of aDDM
    parameters.
    Args:
      choice: integer, either -1 (for left item) or +1 (for right item).
      valueLeft: integer, value of the left item.
      valueRight: integer, value of the right item.
      fixItem: list of items fixated during the trial in chronological order; 1
          correponds to left, 2 corresponds to right, and any other value is
          considered a transition/blank fixation.
      fixTime: list of fixation durations (in miliseconds) in chronological
          order.
      d: float, parameter of the model which controls the speed of integration
          of the signal.
      theta: float between 0 and 1, parameter of the model which controls the
          attentional bias.
      sigma: float, parameter of the model, standard deviation for the normal
          distribution.
      mu: integer, argument to be used as an alternative to sigma, in which case
          sigma = mu * d.
      timeStep: integer, value in miliseconds to be used for binning the time
          axis.
      stateStep: float, to be used for binning the RDV axis.
      barrier: positive number, magnitude of the signal thresholds.
      visualDelay: delay to be discounted from the beginning of all fixations,
          in miliseconds.
      motorDelay: delay to be discounted from the last fixation only, in
          miliseconds.
      plotResults: boolean, flag that determines whether the algorithm evolution
          for the trial should be plotted.
    Returns:
      The likelihood obtained for the given trial and model.
    """

    if sigma == 0:
        if mu != 0:
            sigma = mu * d
        else:
            return 0

    # Iterate over the fixations and discount visual delay.
    if visualDelay > 0:
        correctedFixItem = list()
        correctedFixTime = list()
        for i in xrange(len(fixItem)):
            if fixItem[i] == 1 or fixItem[i] == 2:
                correctedFixItem.append(0)
                correctedFixTime.append(min(visualDelay, fixTime[i]))
                correctedFixItem.append(fixItem[i])
                correctedFixTime.append(max(fixTime[i] - visualDelay, 0))
            else:
                correctedFixItem.append(fixItem[i])
                correctedFixTime.append(fixTime[i])
    else:
        correctedFixItem = list(fixItem)
        correctedFixTime = list(fixTime)

    # Iterate over the fixations and discount motor delay from last fixation.
    if motorDelay > 0:
        for i in xrange(len(correctedFixItem) - 1, -1, -1):
            if correctedFixItem[i] == 1 or correctedFixItem[i] == 2:
                correctedFixTime[i] = max(correctedFixTime[i] - motorDelay, 0)
                break

    # Iterate over the fixations and get the total time for this trial.
    maxTime = 0
    for fTime in correctedFixTime:
        maxTime += int(fTime // timeStep)
    if maxTime == 0:
        return 0
    maxTime += 1

    # The values of the barriers can change over time.
    decay = 0  # decay = 0 means barriers are constant.
    barrierUp = barrier * np.ones(maxTime)
    barrierDown = -barrier * np.ones(maxTime)
    for t in xrange(1, maxTime):
        barrierUp[t] = float(barrier) / float(1 + (decay * t))
        barrierDown[t] = float(-barrier) / float(1 + (decay * t))

    # The vertical axis (RDV space) is divided into states.
    states = np.arange(-barrier, barrier + stateStep, stateStep)
    states[(states < 0.001) & (states > -0.001)] = 0

    # Initial probability for all states is zero, except for the zero state,
    # which has initial probability equal to one.
    prStates = np.zeros(states.size)
    prStates[states == 0] = 1
    prStates[(states > barrierUp[0]) | (states < barrierDown[0])] = 0

    # The probability of crossing each barrier over the time of the trial.
    probUpCrossing = np.zeros(maxTime)
    probDownCrossing = np.zeros(maxTime)

    # Create matrix of traces to keep track of the RDV position probabilities.
    if plotResults:
        traces = np.zeros((states.size, maxTime))
        traces[:, 0] = prStates

    time = 1

    changeMatrix = np.subtract(states.reshape(states.size, 1), states)
    changeUp = np.subtract(barrierUp, states.reshape(states.size, 1))
    changeDown = np.subtract(barrierDown, states.reshape(states.size, 1))

    # Iterate over all fixations in this trial.
    for fItem, fTime in zip(correctedFixItem, correctedFixTime):
        # We use a normal distribution to model changes in RDV stochastically.
        # The mean of the distribution (the change most likely to occur) is
        # calculated from the model parameters and from the item values.
        if fItem == 1:  # Subject is looking left.
            mean = d * (valueLeft - (theta * valueRight))
        elif fItem == 2:  # Subject is looking right.
            mean = d * (-valueRight + (theta * valueLeft))
        else:
            mean = 0

        # Iterate over the time interval of this fixation.
        for t in xrange(int(fTime // timeStep)):
            # Update the probability of the states that remain inside the
            # barriers. The probability of being in state B is the sum, over all
            # states A, of the probability of being in A at the previous
            # timestep times the probability of changing from A to B. We
            # multiply the probability by the stateStep to ensure that the area
            # under the curves for the probability distributions probUpCrossing
            # and probDownCrossing add up to 1.
            prStatesNew = (
                stateStep *
                np.dot(norm.pdf(changeMatrix, mean, sigma), prStates))
            prStatesNew[(states >= barrierUp[time]) |
                        (states <= barrierDown[time])] = 0

            # Calculate the probabilities of crossing the up barrier and the
            # down barrier. This is given by the sum, over all states A, of the
            # probability of being in A at the previous timestep times the
            # probability of crossing the barrier if A is the previous state.
            tempUpCross = np.dot(
                prStates, (1 - norm.cdf(changeUp[:, time], mean, sigma)))
            tempDownCross = np.dot(
                prStates, norm.cdf(changeDown[:, time], mean, sigma))

            # Renormalize to cope with numerical approximations.
            sumIn = np.sum(prStates)
            sumCurrent = np.sum(prStatesNew) + tempUpCross + tempDownCross
            prStatesNew = (prStatesNew * float(sumIn)) / float(sumCurrent)
            tempUpCross = (tempUpCross * float(sumIn)) / float(sumCurrent)
            tempDownCross = (tempDownCross * float(sumIn)) / float(sumCurrent)

            # Update the probabilities of each state and the probabilities of
            # crossing each barrier at this timestep.
            prStates = prStatesNew
            probUpCrossing[time] = tempUpCross
            probDownCrossing[time] = tempDownCross

            # Update traces matrix.
            if plotResults:
                traces[:,time] = prStates

            time += 1

    # Compute the likelihood contribution of this trial based on the final
    # choice.
    likelihood = 0
    if choice == -1:  # Choice was left.
        if probUpCrossing[-1] > 0:
            likelihood = probUpCrossing[-1]
    elif choice == 1:  # Choice was right.
        if probDownCrossing[-1] > 0:
            likelihood = probDownCrossing[-1]

    if plotResults:
        fig1 = plt.figure()
        xAxis = np.arange(0, maxTime * timeStep, timeStep)
        yAxis = np.arange(initialBarrierDown, initialBarrierUp + stateStep,
                          stateStep)
        heatmap = plt.pcolor(xAxis, yAxis, np.flipud(traces))
        plt.xlim(0, maxTime * timeStep - timeStep)
        plt.xlabel('Time')
        plt.ylabel('RDV')
        plt.colorbar(heatmap)

        fig2 = plt.figure()
        plt.plot(range(0, len(probUpCrossing) * timeStep, timeStep),
                 probUpCrossing, label='Up')
        plt.plot(range(0, len(probDownCrossing) * timeStep, timeStep),
                 probDownCrossing, label='Down')
        plt.xlabel('Time')
        plt.ylabel('P(crossing)')
        plt.legend()
        plt.show()

    return likelihood


def get_empirical_distributions(valueLeft, valueRight, fixItem, fixTime,
                                timeStep=10, maxFixTime=3000, numFixDists=3,
                                fixDistType='fixation', useOddTrials=True,
                                useEvenTrials=True, isCisTrial=None,
                                isTransTrial=None, useCisTrials=True,
                                useTransTrials=True):
    """
    Creates empirircal distributions from the data to be used when generating
    model simulations.
    Args:
      valueLeft: dict of dicts, indexed first by subject then by trial number.
          Each entry is an integer corresponding to the value of the left item.
      valueRight: dict of dicts with same indexing as valueLeft. Each entry is
          an integer corresponding to the value of the right item.
      fixItem: dict of dicts with same indexing as valueLeft. Each entry is an
          ordered list of fixated items in the trial.
      fixTime: dict of dicts with same indexing as valueLeft. Each entry is an
          ordered list of fixation durations in the trial.
      timeStep: integer, minimum duration of a fixation to be considered, in
          miliseconds.
      maxFixTime: integer, maximum duration of a fixation to be considered, in
          miliseconds.
      numFixDists: integer, number of fixation types to use in the fixation
          distributions. For instance, if numFixDists equals 3, then 3 separate
          fixation types will be used, corresponding to the 1st, 2nd and other
          (3rd and up) fixations in each trial.
      fixDistType: string, one of {'simple', 'difficulty', 'fixation'}, used to
          determine how the fixation distributions should be indexed. If
          'simple', then fixation distributions will be indexed only by type
          (1st, 2nd, etc). If 'difficulty', they will be indexed by type and by
          trial difficulty. If 'fixation', they will be indexed by type and by
          the value difference between the fixated and unfixated items.
      useOddTrials: boolean, whether or not to use odd trials when creating the
          distributions.
      useEvenTrials: boolean, whether or not to use even trials when creating
          the distributions.
      isCisTrial: dict of dicts, indexed first by subject then by trial number.
          Applies to perceptual decisions only. Each entry is a boolean
          indicating if the trial is cis (both bars on the same side of the
          target).
      isTransTrial: dict of dicts with same indexing as isCisTrial. Applies to
          perceptual decisions only. Each entry is a boolean indicating if the
          trial is trans (bars on either side of the target).
      useCisTrials: boolean, whether or not to use cis trials when creating the
          distributions (for perceptual decisions only).
      useTransTrials: boolean, whether or not to use trans trials when creating
          the distributions (for perceptual decisions only).
    Returns:
      A named tuple containing the following fields:
        probLeftFixFirst: float between 0 and 1, empirical probability that the
            left item will be fixated first.
        distLatencies: numpy array corresponding to the empirical distribution
            of trial latencies (delay before first fixation) in miliseconds.
        distTransitions: numpy array corresponding to the empirical distribution
            of transitions (delays between item fixations) in miliseconds.
        distFixations: dict whose indexing is controlled by argument
            fixDistType. Its entries are numpy arrays corresponding to the
            empirical distributions of item fixation durations in miliseconds.
    """

    if fixDistType == 'difficulty':
        valueDiffs = range(0,4,1)
    elif fixDistType == 'fixation':
        valueDiffs = range(-3,4,1)

    countLeftFirst = 0
    countTotalTrials = 0
    distLatenciesList = list()
    distTransitionsList = list()
    distFixationsList = dict()
    for fixNumber in xrange(1, numFixDists + 1):
        if fixDistType == 'simple':
            distFixationsList[fixNumber] = list()
        else:
            distFixationsList[fixNumber] = dict()
            for valueDiff in valueDiffs:
                distFixationsList[fixNumber][valueDiff] = list()

    subjects = valueLeft.keys()
    for subject in subjects:
        trials = valueLeft[subject].keys()
        for trial in trials:
            if not useOddTrials and trial % 2 != 0:
                continue
            if not useEvenTrials and trial % 2 == 0:
                continue
            if (not useCisTrials and isCisTrial[subject][trial] and
                not isTransTrial[subject][trial]):
                continue
            if (not useTransTrials and isTransTrial[subject][trial] and
                not isCisTrial[subject][trial]):
                continue

            # Discard trial if it has 1 or less item fixations.
            items = fixItem[subject][trial]
            if items[(items==1) | (items==2)].shape[0] <= 1:
                continue
            fixUnfixValueDiffs = {
                1: valueLeft[subject][trial] - valueRight[subject][trial],
                2: valueRight[subject][trial] - valueLeft[subject][trial]}
            # Find the last item fixation in this trial.
            excludeCount = 0
            for i in xrange(fixItem[subject][trial].shape[0] - 1, -1, -1):
                excludeCount += 1
                if (fixItem[subject][trial][i] == 1 or
                    fixItem[subject][trial][i] == 2):
                    break
            # Iterate over this trial's fixations (skip the last item fixation).
            latency = 0
            firstItemFixReached = False
            fixNumber = 1
            for i in xrange(fixItem[subject][trial].shape[0] - excludeCount):
                item = fixItem[subject][trial][i]
                if item != 1 and item != 2:
                    if not firstItemFixReached:
                        latency += fixTime[subject][trial][i]
                    elif (fixTime[subject][trial][i] >= timeStep and
                          fixTime[subject][trial][i] <= maxFixTime):
                        distTransitionsList.append(fixTime[subject][trial][i])
                else:
                    if not firstItemFixReached:
                        firstItemFixReached = True
                        distLatenciesList.append(latency)
                    if fixNumber == 1:
                        countTotalTrials +=1
                        if item == 1:  # First fixation was left.
                            countLeftFirst += 1
                    if (fixTime[subject][trial][i] >= timeStep and
                        fixTime[subject][trial][i] <= maxFixTime):
                        if fixDistType == 'simple':
                            distFixationsList[fixNumber].append(
                                fixTime[subject][trial][i])
                        elif fixDistType == 'difficulty':
                            valueDiff = np.absolute(
                                valueLeft[subject][trial] -
                                valueRight[subject][trial])
                            distFixationsList[fixNumber][valueDiff].append(
                                fixTime[subject][trial][i])
                        elif fixDistType == 'fixation':
                            valueDiff = fixUnfixValueDiffs[item]
                            distFixationsList[fixNumber][valueDiff].append(
                                fixTime[subject][trial][i])
                    if fixNumber < numFixDists:
                        fixNumber += 1

    probLeftFixFirst = float(countLeftFirst) / float(countTotalTrials)
    distLatencies = np.array(distLatenciesList)
    distTransitions = np.array(distTransitionsList)
    distFixations = dict()
    for fixNumber in xrange(1, numFixDists + 1):
        if fixDistType == 'simple':
            distFixations[fixNumber] = np.array(distFixationsList[fixNumber])
        else:
            distFixations[fixNumber] = dict()
            for valueDiff in valueDiffs:
                distFixations[fixNumber][valueDiff] = np.array(
                    distFixationsList[fixNumber][valueDiff])

    dists = collections.namedtuple(
        'Dists', ['probLeftFixFirst', 'distLatencies', 'distTransitions',
        'distFixations'])
    return dists(probLeftFixFirst, distLatencies, distTransitions,
                 distFixations)


def run_simulations(probLeftFixFirst, distLatencies, distTransitions,
                    distFixations, numTrials, trialConditions, d, theta,
                    sigma=0, mu=0, timeStep=10, barrier=1, numFixDists=3,
                    fixDistType='fixation', visualDelay=0, motorDelay=0):
    """
    Generates aDDM simulations given the model parameters and some empirical
    fixation data, which are used to generate the simulated fixations.
    Args:
      probLeftFixFirst: float between 0 and 1, empirical probability that the
          left item will be fixated first.
      distLatencies: numpy array corresponding to the empirical distribution of
          trial latencies (delay before first fixation) in miliseconds.
      distTransitions: numpy array corresponding to the empirical distribution
          of transitions (delays between item fixations) in miliseconds.
      distFixations: dict whose indexing is controlled by argument fixDistType.
          Its entries are numpy arrays corresponding to the empirical
          distributions of item fixation durations in miliseconds.
      numTrials: integer, number of simulations to be generated for each trial
          condition.
      trialConditions: list of tuples, where each entry is a pair (valueLeft,
          valueRight), containing the values of the two items.
      d: float, parameter of the model which controls the speed of integration
          of the signal.
      theta: float between 0 and 1, parameter of the model which controls the
          attentional bias.
      sigma: float, parameter of the model, standard deviation for the normal
          distribution.
      mu: integer, argument to be used as an alternative to sigma, in which case
          sigma = mu * d.
      timeStep: integer, value in miliseconds to be used for binning the time
          axis.
      barrier: positive number, magnitude of the signal thresholds.
      numFixDists: integer, number of fixation types to use in the fixation
          distributions. For instance, if numFixDists equals 3, then 3 separate
          fixation types will be used, corresponding to the 1st, 2nd and other
          (3rd and up) fixations in each trial.
      fixDistType: string, one of {'simple', 'difficulty', 'fixation'},
          determines how the fixation distributions are indexed. If 'simple',
          fixation distributions are indexed only by type (1st, 2nd, etc). If
          'difficulty', they are indexed by type and by trial difficulty. If
          'fixation', they are indexed by type and by the value difference
          between the fixated and unfixated items.
      visualDelay: delay to be discounted from the beginning of all fixations,
          in miliseconds.
      motorDelay: delay to be discounted from the last fixation only, in
          miliseconds.
    Returns:
      A named tuple containing the following fields:
        RT: dict indexed by trial number, where each entry corresponds to the
            reaction time in miliseconds.
        choice: dict indexed by trial number, where each entry is either -1 (for
            left item) or +1 (for right item).
        valueLeft: dict indexed by trial number, where each entry corresponds to
            the value of the left item.
        valueRight: dict indexed by trial number, where each entry corresponds
            to the value of the right item.
        fixItem: dict indexed by trial number, where each entry is an ordered
            list of fixated items in the trial: 1 is used for left, 2 for right
            and 0 for latencies/transitions.
        fixTime: dict indexed by trial number, where each entry is an ordered
            list of fixation durations in miliseconds.
        fixRDV: dict indexed by trual number, where each entry is a list of
            floats corresponding to the RDV values at the end of each fixation
            in the trial.
        uninterruptedLastFixTime: dict indexed by trial number, where each entry
            is an integer corresponding to the duration, in miliseconds, that
            the last fixation in the trial would have had it not been
            interrupted when a decision was made.
    """

    if sigma == 0:
        if mu != 0:
            sigma = mu * d
        else:
            return None

    # Simulation data to be returned.
    RT = dict()
    choice = dict()
    valueLeft = dict()
    valueRight = dict()
    fixItem = dict()
    fixTime = dict()
    fixRDV = dict()
    uninterruptedLastFixTime = dict()

    trialCount = 0

    for trialCondition in trialConditions:
        vLeft = trialCondition[0]
        vRight = trialCondition[1]
        fixUnfixValueDiffs = {1: vLeft - vRight, 2: vRight - vLeft}
        trial = 0
        while trial < numTrials:
            fixItem[trialCount] = list()
            fixTime[trialCount] = list()
            fixRDV[trialCount] = list()

            RDV = 0
            trialTime = 0

            # Sample and iterate over the latency for this trial.
            trialAborted = False
            while True:
                latency = np.random.choice(distLatencies)
                for t in xrange(int(latency // timeStep)):
                    # Sample the change in RDV from the distribution.
                    RDV += np.random.normal(0, sigma)
                    # If the RDV hit one of the barriers, we abort the trial,
                    # since a trial must end on an item fixation.
                    if RDV >= barrier or RDV <= -barrier:
                        trialAborted = True
                        break

                if trialAborted:
                    RDV = 0
                    trialAborted = False
                    continue
                else:
                    # Add latency to this trial's data.
                    fixRDV[trialCount].append(RDV)
                    fixItem[trialCount].append(0)
                    fixTime[trialCount].append(latency - (latency % timeStep))
                    trialTime += latency - (latency % timeStep)
                    break

            # Sample the first fixation for this trial.
            probLeftRight = np.array([probLeftFixFirst, 1-probLeftFixFirst])
            currFixItem = np.random.choice([1, 2], p=probLeftRight)
            if fixDistType == 'simple':
                currFixTime = np.random.choice(distFixations[1]) - visualDelay
            elif fixDistType == 'difficulty':
                valueDiff = np.absolute(vLeft - vRight)
                currFixTime = (
                    np.random.choice(distFixations[1][valueDiff]) - visualDelay)
            elif fixDistType == 'fixation':
                valueDiff = fixUnfixValueDiffs[currFixItem]
                currFixTime = (
                    np.random.choice(distFixations[1][valueDiff]) - visualDelay)

            # Iterate over all fixations in this trial.
            fixNumber = 2
            trialFinished = False
            trialAborted = False
            while True:
                # Iterate over the visual delay for the current fixation.
                for t in xrange(int(visualDelay // timeStep)):
                    # Sample the change in RDV from the distribution.
                    RDV += np.random.normal(0, sigma)

                    # If the RDV hit one of the barriers, the trial is over.
                    if RDV >= barrier or RDV <= -barrier:
                        if RDV >= barrier:
                            choice[trialCount] = -1
                        elif RDV <= -barrier:
                            choice[trialCount] = 1
                        valueLeft[trialCount] = vLeft
                        valueRight[trialCount] = vRight
                        fixRDV[trialCount].append(RDV)
                        fixItem[trialCount].append(currFixItem)
                        fixTime[trialCount].append(
                            ((t + 1) * timeStep) + motorDelay)
                        trialTime += ((t + 1) * timeStep) + motorDelay
                        RT[trialCount] = trialTime
                        uninterruptedLastFixTime[trialCount] = currFixTime
                        trialFinished = True
                        break

                if trialFinished:
                    break

                # Iterate over the time interval of the current fixation.
                for t in xrange(int(currFixTime // timeStep)):
                    # We use a distribution to model changes in RDV
                    # stochastically. The mean of the distribution (the change
                    # most likely to occur) is calculated from the model
                    # parameters and from the values of the two items.
                    if currFixItem == 1:  # Subject is looking left.
                        mean = d * (vLeft - (theta * vRight))
                    elif currFixItem == 2:  # Subject is looking right.
                        mean = d * (-vRight + (theta * vLeft))

                    # Sample the change in RDV from the distribution.
                    RDV += np.random.normal(mean, sigma)

                    # If the RDV hit one of the barriers, the trial is over.
                    if RDV >= barrier or RDV <= -barrier:
                        if RDV >= barrier:
                            choice[trialCount] = -1
                        elif RDV <= -barrier:
                            choice[trialCount] = 1
                        valueLeft[trialCount] = vLeft
                        valueRight[trialCount] = vRight
                        fixRDV[trialCount].append(RDV)
                        fixItem[trialCount].append(currFixItem)
                        fixTime[trialCount].append(
                            ((t + 1) * timeStep) + visualDelay + motorDelay)
                        trialTime += (((t + 1) * timeStep) +
                                      visualDelay + motorDelay)
                        RT[trialCount] = trialTime
                        uninterruptedLastFixTime[trialCount] = currFixTime
                        trialFinished = True
                        break

                if trialFinished:
                    break

                # Add previous fixation to this trial's data.
                fixRDV[trialCount].append(RDV)
                fixItem[trialCount].append(currFixItem)
                fixTime[trialCount].append(
                    (currFixTime - (currFixTime % timeStep)) + visualDelay)
                trialTime += ((currFixTime - (currFixTime % timeStep)) +
                              visualDelay)

                # Sample and iterate over transition time.
                transitionTime = np.random.choice(distTransitions)
                for t in xrange(int(transitionTime // timeStep)):
                    # Sample the change in RDV from the distribution.
                    RDV += np.random.normal(0, sigma)

                    # If the RDV hit one of the barriers, we abort the trial,
                    # since a trial must end on an item fixation.
                    if RDV >= barrier or RDV <= -barrier:
                        trialFinished = True
                        trialAborted = True
                        break

                if trialFinished:
                    break

                # Add previous transition to this trial's data.
                fixRDV[trialCount].append(RDV)
                fixItem[trialCount].append(0)
                fixTime[trialCount].append(
                    transitionTime - (transitionTime % timeStep))
                trialTime += transitionTime - (transitionTime % timeStep)

                # Sample the next fixation for this trial.
                if currFixItem == 1:
                    currFixItem = 2
                elif currFixItem == 2:
                    currFixItem = 1
                if fixDistType == 'simple':
                    currFixTime = (np.random.choice(distFixations[fixNumber]) -
                                   visualDelay)
                elif fixDistType == 'difficulty':
                    valueDiff = np.absolute(vLeft - vRight)
                    currFixTime = np.random.choice(
                        distFixations[fixNumber][valueDiff]) - visualDelay
                elif fixDistType == 'fixation':
                    valueDiff = fixUnfixValueDiffs[currFixItem]
                    currFixTime = np.random.choice(
                        distFixations[fixNumber][valueDiff]) - visualDelay
                if fixNumber < numFixDists:
                    fixNumber += 1

            # Move on to the next trial.
            if not trialAborted:
                trial += 1
                trialCount += 1

    simul = collections.namedtuple(
        'Simul', ['RT', 'choice', 'valueLeft', 'valueRight', 'fixItem',
        'fixTime', 'fixRDV', 'uninterruptedLastFixTime'])
    return simul(RT, choice, valueLeft, valueRight, fixItem, fixTime, fixRDV,
                 uninterruptedLastFixTime)


def generate_probabilistic_simulations(probLeftFixFirst, distLatencies, 
                                       distTransitions, distFixations,
                                       trialConditions, posteriors,
                                       numSamples=100,
                                       numSimulationsPerSample=10, timeStep=10,
                                       barrier=1, numFixDists=3,
                                       fixDistType='fixation', visualDelay=0,
                                       motorDelay=0):
    """
    Generates aDDM simulations given a posterior distribution over a grid of
    model parameters and some empirical fixation data, which are used to
    generate the simulated fixations.
    Args:
      probLeftFixFirst: float between 0 and 1, empirical probability that the
          left item will be fixated first.
      distLatencies: numpy array corresponding to the empirical distribution of
          trial latencies (delay before first fixation) in miliseconds.
      distTransitions: numpy array corresponding to the empirical distribution
          of transitions (delays between item fixations) in miliseconds.
      distFixations: dict whose indexing is controlled by argument fixDistType.
          Its entries are numpy arrays corresponding to the empirical
          distributions of item fixation durations in miliseconds.
      numTrials: integer, number of simulations to be generated for each trial
          condition.
      trialConditions: list of tuples, where each entry is a pair (valueLeft,
          valueRight), containing the values of the two items.
      posteriors: dict indexed by model, where a model is a tuple containing the
          3 parameters of the model. Each entry is a float between 0 and 1
          corresponding to the posterior probability of the model.
      numSamples: integer, number of times a model should be sampled from the
          posteriors distribution.
      numSimulationsPerSample: integer, number of simulations to be generated
          for each sampled model.
      timeStep: integer, value in miliseconds to be used for binning the time
          axis.
      barrier: positive number, magnitude of the signal thresholds.
      numFixDists: integer, number of fixation types to use in the fixation
          distributions. For instance, if numFixDists equals 3, then 3 separate
          fixation types will be used, corresponding to the 1st, 2nd and other
          (3rd and up) fixations in each trial.
      fixDistType: string, one of {'simple', 'difficulty', 'fixation'},
          determines how the fixation distributions are indexed. If 'simple',
          fixation distributions are indexed only by type (1st, 2nd, etc). If
          'difficulty', they are indexed by type and by trial difficulty. If
          'fixation', they are indexed by type and by the value difference
          between the fixated and unfixated items.
      visualDelay: delay to be discounted from the beginning of all fixations,
          in miliseconds.
      motorDelay: delay to be discounted from the last fixation only, in
          miliseconds.
    Returns:
      A named tuple containing the following fields:
        RT: dict indexed by trial number, where each entry corresponds to the
            reaction time in miliseconds.
        choice: dict indexed by trial number, where each entry is either -1 (for
            left item) or +1 (for right item).
        valueLeft: dict indexed by trial number, where each entry corresponds to
            the value of the left item.
        valueRight: dict indexed by trial number, where each entry corresponds
            to the value of the right item.
        fixItem: dict indexed by trial number, where each entry is an ordered
            list of fixated items in the trial: 1 is used for left, 2 for right
            and 0 for latencies/transitions.
        fixTime: dict indexed by trial number, where each entry is an ordered
            list of fixation durations in miliseconds.
        fixRDV: dict indexed by trual number, where each entry is a list of
            floats corresponding to the RDV values at the end of each fixation
            in the trial.
        uninterruptedLastFixTime: dict indexed by trial number, where each entry
            is an integer corresponding to the duration, in miliseconds, that
            the last fixation in the trial would have had it not been
            interrupted when a decision was made.
    """

    posteriorsList = list()
    models = dict()
    i = 0
    for model, posterior in posteriors.iteritems():
        posteriorsList.append(posterior)
        models[i] = model
        i += 1

    RT = dict()
    choice = dict()
    valueLeft = dict()
    valueRight = dict()
    fixItem = dict()
    fixTime = dict()
    fixRDV = dict()
    uninterruptedLastFixTime = dict()

    numModels = len(models.keys())
    trialCount = 0
    for i in xrange(numSamples):
        # Sample model from posteriors distribution.
        modelIndex = np.random.choice(
            np.array(range(numModels)), p=np.array(posteriorsList))
        model = models[modelIndex]
        d = model[0]
        theta = model[1]
        sigma = model[2]

        # Generate simulations with the sampled model.
        simul = run_simulations(
            probLeftFixFirst, distLatencies, distTransitions, distFixations,
            numSimulationsPerSample, trialConditions, d, theta, sigma=sigma,
            mu=0, timeStep=10, barrier=1, numFixDists=3, fixDistType='fixation',
            visualDelay=0, motorDelay=0)
        for trial in simul.RT.keys():
            RT[trialCount] = simul.RT[trial]
            choice[trialCount] = simul.choice[trial]
            valueLeft[trialCount] = simul.valueLeft[trial]
            valueRight[trialCount] = simul.valueRight[trial]
            fixTime[trialCount] = simul.fixTime[trial]
            fixItem[trialCount] = simul.fixItem[trial]
            fixRDV[trialCount] = simul.fixRDV[trial]
            uninterruptedLastFixTime[trialCount] = (
                simul.uninterruptedLastFixTime[trial])
            
            trialCount += 1

    simul = collections.namedtuple(
        'Simul', ['RT', 'choice', 'valueLeft', 'valueRight', 'fixItem',
        'fixTime', 'fixRDV', 'uninterruptedLastFixTime'])
    return simul(RT, choice, valueLeft, valueRight, fixItem, fixTime, fixRDV,
                 uninterruptedLastFixTime)
