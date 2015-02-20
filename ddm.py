#!/usr/bin/python

# ddm.py
# Author: Gabriela Tavares, gtavares@caltech.edu

from multiprocessing import Pool
from scipy.stats import norm

import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def analysis_per_trial(rt, choice, valueLeft, valueRight, d, std, timeStep=10,
    barrier=1, plotResults=False):
    stateStep = 0.1

    # Get the total time for this trial.
    maxTime = int(rt // timeStep)

    # The values of the barriers can change over time.
    decay = 0  # decay = 0 means barriers are constant.
    barrierUp = barrier * np.ones(maxTime)
    barrierDown = -barrier * np.ones(maxTime)
    for t in xrange(maxTime):
        barrierUp[t] = float(barrier) / float(1+decay*(t+1))
        barrierDown[t] = float(-barrier) / float(1+decay*(t+1))

    # The vertical axis (RDV space) is divided into states.
    states = np.arange(-barrier, barrier + stateStep, stateStep)
    idx = np.where(np.logical_and(states<0.01, states>-0.01))[0]
    states[idx] = 0

    # Initial probability for all states is zero, except for the zero state,
    # which has initial probability equal to one.
    prStates = np.zeros(states.size)
    idx = np.where(states==0)[0]
    prStates[idx] = 1

    # The probability of crossing each barrier over the time of the trial.
    probUpCrossing = np.zeros(maxTime)
    probDownCrossing = np.zeros(maxTime)

    # Create matrix of traces to keep track of the RDV position probabilities.
    if plotResults:
        traces = np.zeros((states.size, maxTime))
        traces[:, 0] = prStates

    # We use a normal distribution to model changes in RDV stochastically.
    # The mean of the distribution (the change most likely to occur) is
    # calculated from the model parameter d and from the item values.
    mean = d * (valueLeft - valueRight)

    # Iterate over the time of this trial.
    for time in xrange(maxTime):
        prStatesNew = np.zeros(states.size)

        # Update the probability of the states that remain inside the
        # barriers.
        for s in xrange(0, states.size):
            currState = states[s]
            if currState > barrierDown[time] and currState < barrierUp[time]:
                change = (currState * np.ones(states.size)) - states
                # The probability of being in state B is the sum, over all
                # states A, of the probability of being in A at the previous
                # timestep times the probability of changing from A to B.
                # We multiply the probability by the stateStep to ensure
                # that the area under the curve for the probability
                # distributions probUpCrossing and probDownCrossing each add
                # up to 1.
                prStatesNew[s] = (stateStep * np.sum(np.multiply(prStates,
                    norm.pdf(change, mean, std))))

        # Calculate the probabilities of crossing the up barrier and the
        # down barrier. This is given by the sum, over all states A, of the
        # probability of being in A at the previous timestep times the
        # probability of crossing the barrier if A is the previous state.
        changeUp = (barrierUp[time] * np.ones(states.size)) - states
        tempUpCross = np.sum(np.multiply(prStates,
            (1 - norm.cdf(changeUp, mean, std))))
        changeDown = (barrierDown[time] * np.ones(states.size)) - states
        tempDownCross = np.sum(np.multiply(prStates,
            (norm.cdf(changeDown, mean, std))))

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


def run_simulations(numTrials, trialConditions, d, std, timeStep=10, barrier=1):
    # Simulation data to be returned.
    rt = dict()
    choice = dict()
    valueLeft = dict()
    valueRight = dict()

    trialCount = 0

    for trialCondition in trialConditions:
        vLeft = trialCondition[0]
        vRight = trialCondition[1]
        mean = d * (vLeft - vRight)

        for trial in xrange(numTrials):
            RDV = 0
            time = 0
            while True:
                # If the RDV hit one of the barriers, the trial is over.
                if RDV >= barrier or RDV <= -barrier:
                    rt[trialCount] = time * timeStep
                    valueLeft[trialCount] = vLeft
                    valueRight[trialCount] = vRight
                    if RDV >= barrier:
                        choice[trialCount] = -1
                    elif RDV <= -barrier:
                        choice[trialCount] = 1
                    break

                # Sample the change in RDV from the distribution.
                RDV += np.random.normal(mean, std)

                time += 1

            # Move on to the next trial.
            trialCount += 1

    simul = collections.namedtuple('Simul', ['rt', 'choice', 'valueLeft',
        'valueRight'])
    return simul(rt, choice, valueLeft, valueRight)


def run_analysis_wrapper(params):
    return analysis_per_trial(*params)


def main():
    numThreads = 9
    pool = Pool(numThreads)

    # Parameters for generating simulations.
    d = 0.006
    std = 0.08
    numTrials = 20
    numValues = 4
    values = range(0,numValues,1)
    trialConditions = list()
    for vLeft in values:
        for vRight in values:
            trialConditions.append((vLeft, vRight))

    # Generate simulations.
    simul = run_simulations(numTrials, trialConditions, d, std)
    rt = simul.rt
    choice = simul.choice
    valueLeft = simul.valueLeft
    valueRight = simul.valueRight

    rangeD = [0.004, 0.006, 0.008]
    rangeStd = [0.07, 0.08, 0.09]
    numModels = len(rangeD) * len(rangeStd)

    models = list()
    posteriors = dict()
    for d in rangeD:
        for std in rangeStd:
            model = (d, std)
            models.append(model)
            posteriors[model] = 1./ numModels

    trials = rt.keys()
    for trial in trials:
        listParams = list()
        for model in models:
            listParams.append((rt[trial], choice[trial], valueLeft[trial],
                valueRight[trial], model[0], model[1]))
        likelihoods = pool.map(run_analysis_wrapper, listParams)

        # Get the denominator for normalizing the posteriors.
        i = 0
        denominator = 0
        for model in models:
            denominator += posteriors[model] * likelihoods[i]
            i += 1
        if denominator == 0:
            continue

        # Calculate the posteriors after this trial.
        i = 0
        for model in models:
            if likelihoods[i] != 0:
                prior = posteriors[model]
                posteriors[model] = likelihoods[i] * prior / denominator
            i += 1

    print("Number of trials: " + str(numTrials))
    print("Number of values: " + str(numValues))
    for model in posteriors:
        print("P" + str(model) + " = " + str(posteriors[model]))
    print("Sum: " + str(sum(posteriors.values())))


if __name__ == '__main__':
    main()