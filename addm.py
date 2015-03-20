#!/usr/bin/python

# handle_fixations.py
# Author: Gabriela Tavares, gtavares@caltech.edu

from scipy.stats import norm

import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data_from_csv(expdataFile, fixationsFile):
    # Load experimental data from CSV file.
    # Format: parcode, trial, rt, choice, val_left, val_right.
    # Angular distances to target are transformed to values in [0, 1, 2, 3].
    df = pd.DataFrame.from_csv(expdataFile, header=0, sep=',', index_col=None)
    subjects = df.parcode.unique()

    rt = dict()
    choice = dict()
    distLeft = dict()
    distRight = dict()

    for subject in subjects:
        rt[subject] = dict()
        choice[subject] = dict()
        distLeft[subject] = dict()
        distRight[subject] = dict()
        dataSubject = np.array(df.loc[df['parcode']==subject,
            ['trial','rt','choice','val_left','val_right']])
        trials = np.unique(dataSubject[:,0]).tolist()
        for trial in trials:
            dataTrial = np.array(df.loc[(df['trial']==trial) &
                (df['parcode']==subject), ['rt','choice','val_left',
                'val_right']])
            rt[subject][trial] = dataTrial[0,0]
            choice[subject][trial] = dataTrial[0,1]
            distLeft[subject][trial] = dataTrial[0,2]
            distRight[subject][trial] = dataTrial[0,3]

    # Load fixation data from CSV file.
    # Format: parcode, trial, fix_item, fix_time.
    df = pd.DataFrame.from_csv(fixationsFile, header=0, sep=',',
        index_col=None)
    subjects = df.parcode.unique()

    fixItem = dict()
    fixTime = dict()

    for subject in subjects:
        fixItem[subject] = dict()
        fixTime[subject] = dict()
        dataSubject = np.array(df.loc[df['parcode']==subject,
            ['trial','fix_item','fix_time']])
        trials = np.unique(dataSubject[:,0]).tolist()
        for trial in trials:
            dataTrial = np.array(df.loc[(df['trial']==trial) &
                (df['parcode']==subject), ['fix_item','fix_time']])
            fixItem[subject][trial] = dataTrial[:,0]
            fixTime[subject][trial] = dataTrial[:,1]

    data = collections.namedtuple('Data', ['rt', 'choice', 'distLeft',
        'distRight', 'fixItem', 'fixTime'])
    return data(rt, choice, distLeft, distRight, fixItem, fixTime)


def analysis_per_trial(rt, choice, valueLeft, valueRight, fixItem, fixTime, d,
    theta, std=0, mu=0, timeStep=10, barrier=1, visualDelay=0, motorDelay=0,
    plotResults=False, nonFixDiffusion=True):
    stateStep = 0.1
    if std == 0:
        if mu != 0:
            std = mu * d
        else:
            return 0

    # Iterate over the fixations and discount visual delay.
    for i in xrange(len(fixItem)):
        if fixItem[i] == 1 or fixItem[i] == 2:
            fixTime[i] = max(fixTime[i] - visualDelay, 0)
            fixItem = np.append(fixItem, 0)
            fixTime = np.append(fixTime, visualDelay)

    # Iterate over the fixations and discount motor delay from last fixation.
    for i in xrange(len(fixItem) - 1, -1, -1):
        if fixItem[i] == 1 or fixItem[i] == 2:
            fixTime[i] = max(fixTime[i] - motorDelay, 0)
            break

    # Iterate over the fixations and get the total time for this trial.
    maxTime = 0
    for fItem, fTime in zip(fixItem, fixTime):
        maxTime += int(fTime // timeStep)
    if maxTime == 0:
        return 0

    # The values of the barriers can change over time.
    decay = 0  # decay = 0 means barriers are constant.
    barrierUp = barrier * np.ones(maxTime)
    barrierDown = -barrier * np.ones(maxTime)
    for t in xrange(0, int(maxTime)):
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

    time = 0

    # Iterate over all fixations in this trial.
    for fItem, fTime in zip(fixItem, fixTime):
        # We use a normal distribution to model changes in RDV stochastically.
        # The mean of the distribution (the change most likely to occur) is
        # calculated from the model parameters and from the item values.
        if fItem == 1:  # Subject is looking left.
            mean = d * (valueLeft - (theta * valueRight))
        elif fItem == 2:  # Subject is looking right.
            mean = d * (-valueRight + (theta * valueLeft))
        else:
            if nonFixDiffusion == True:
                #print('Setting non-fix diffusion rate')  Debugging message
                mean = d * (valueLeft - valueRight)
            else:
                mean = 0

        # Iterate over the time interval of this fixation.
        for t in xrange(int(fTime // timeStep)):
            prStatesNew = np.zeros(states.size)

            # Update the probability of the states that remain inside the
            # barriers.
            for s in xrange(0, states.size):
                currState = states[s]
                if (currState > barrierDown[time] and
                    currState < barrierUp[time]):
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


def get_empirical_distributions(rt, choice, valueLeft, valueRight, fixItem,
    fixTime, useOddTrials=True, useEvenTrials=True, useCisTrials=True,
    useTransTrials=True):
    valueDiffs = range(0,5,1)

    countLeftFirst = 0
    countTotalTrials = 0
    distTransitionList = list()
    distFirstFixList = dict()
    distSecondFixList = dict()
    distThirdFixList = dict()
    distOtherFixList = dict()
    for valueDiff in valueDiffs:
        distFirstFixList[valueDiff] = list()
        distSecondFixList[valueDiff] = list()
        distThirdFixList[valueDiff] = list()
        distOtherFixList[valueDiff] = list()

#    # Get item values.
#    valueLeft = dict()
#    valueRight = dict()
#    subjects = distLeft.keys()
#    for subject in subjects:
#        valueLeft[subject] = dict()
#        valueRight[subject] = dict()
#        trials = distLeft[subject].keys()
#        for trial in trials:
#            valueLeft[subject][trial] = np.absolute((np.absolute(
#                distLeft[subject][trial])-15)/5)
#            valueRight[subject][trial] = np.absolute((np.absolute(
#                distRight[subject][trial])-15)/5)

    subjects = rt.keys()
    for subject in subjects:
        trials = rt[subject].keys()
        for trial in trials:
            if not useOddTrials and trial % 2 != 0:
                continue
            if not useEvenTrials and trial % 2 == 0:
                continue
#            if (not useCisTrials and (distLeft[subject][trial] *
#                distRight[subject][trial] > 0)):
#                continue
#            if (not useTransTrials and (distLeft[subject][trial] *
#                distRight[subject][trial] < 0)):
#                continue
            if fixItem[subject][trial].shape[0] < 2:
                continue
            # Discard trial if it has 1 or less item fixations.
            items = fixItem[subject][trial]
            if items[(items==1) | (items==2)].shape[0] <= 1:
                continue
            # Get value difference between best and worst items for this trial.
            valueDiff = np.absolute(valueLeft[subject][trial] -
                valueRight[subject][trial])
            # Find the last item fixation in this trial.
            excludeCount = 0
            for i in xrange(fixItem[subject][trial].shape[0] - 1, -1, -1):
                excludeCount += 1
                if (fixItem[subject][trial][i] == 1 or
                    fixItem[subject][trial][i] == 2):
                    break
            # Iterate over this trial's fixations (skip the last item fixation).
            fixNumber = 1
            for i in xrange(fixItem[subject][trial].shape[0] - excludeCount):
                item = fixItem[subject][trial][i]
                if item != 1 and item != 2:
                    distTransitionList.append(fixTime[subject][trial][i])
                else:
                    if fixNumber == 1:
                        fixNumber += 1
                        if fixTime[subject][trial][i] > 0:
                            distFirstFixList[valueDiff].append(
                                fixTime[subject][trial][i])
                        countTotalTrials +=1
                        if item == 1:  # First fixation was left.
                            countLeftFirst += 1
                    elif fixNumber == 2:
                        fixNumber += 1
                        if fixTime[subject][trial][i] > 0:
                            distSecondFixList[valueDiff].append(
                                fixTime[subject][trial][i])
                    elif fixNumber == 3:
                        fixNumber += 1
                        if fixTime[subject][trial][i] > 0:
                            distThirdFixList[valueDiff].append(
                                fixTime[subject][trial][i])
                    else:
                        if fixTime[subject][trial][i] > 0:
                            distOtherFixList[valueDiff].append(
                                fixTime[subject][trial][i])

    probLeftFixFirst = float(countLeftFirst) / float(countTotalTrials)
    distTransition = np.array(distTransitionList)
    distFirstFix = dict()
    distSecondFix = dict()
    distThirdFix = dict()
    distOtherFix = dict()
    for valueDiff in valueDiffs:
        distFirstFix[valueDiff] = np.array(distFirstFixList[valueDiff])
        distSecondFix[valueDiff] = np.array(distSecondFixList[valueDiff])
        distThirdFix[valueDiff] = np.array(distThirdFixList[valueDiff])
        distOtherFix[valueDiff] = np.array(distOtherFixList[valueDiff])

    dists = collections.namedtuple('Dists', ['probLeftFixFirst',
        'distTransition', 'distFirstFix', 'distSecondFix', 'distThirdFix',
        'distOtherFix'])
    return dists(probLeftFixFirst, distTransition, distFirstFix, distSecondFix,
        distThirdFix, distOtherFix)


def run_simulations(probLeftFixFirst, distTransition, distFirstFix,
    distSecondFix, distThirdFix, distOtherFix, numTrials, trialConditions, d,
    theta, std=0, mu=0, timeStep=10, barrier=1, visualDelay=0, motorDelay=0,
    nonFixDiffusion = True):
    if std == 0:
        if mu != 0:
            std = mu * d
        else:
            return None

    # Simulation data to be returned.
    rt = dict()
    choice = dict()
    valLeft = dict()
    valRight = dict()
    fixItem = dict()
    fixTime = dict()
    fixRDV = dict()

    trialCount = 0

    for trialCondition in trialConditions:
        vLeft = trialCondition[0]
        vRight = trialCondition[1]
        valueDiff = np.absolute(vLeft - vRight)
        trial = 0
        while trial < numTrials:
            fixItem[trialCount] = list()
            fixTime[trialCount] = list()
            fixRDV[trialCount] = list()

            # Sample the first fixation for this trial.
            probLeftRight = np.array([probLeftFixFirst, 1-probLeftFixFirst])
            currFixItem = np.random.choice([1, 2], p=probLeftRight)
            currFixTime = (np.random.choice(distFirstFix[valueDiff]) -
                visualDelay)
            
            fixNumber = 2  # This is set to 2, because we have just sampled the first fixation
            
            # Iterate over all fixations in this trial.
           
            trialFinished = False
            trialAborted = False
            RDV = 0
            trialTime = 0
            while True:
                # Iterate over the visual delay for the current fixation.
                currRDV = RDV
                for t in xrange(int(visualDelay // timeStep)):
                    # If the RDV hit one of the barriers, the trial is over.
                    if RDV >= barrier or RDV <= -barrier:
                        if RDV >= barrier:
                            choice[trialCount] = -1
                        elif RDV <= -barrier:
                            choice[trialCount] = 1
                        valLeft[trialCount] = trialCondition[0]
                        valRight[trialCount] = trialCondition[1]
                        fixRDV[trialCount].append(currRDV)
                        fixItem[trialCount].append(currFixItem)
                        fixTime[trialCount].append((t * timeStep) + motorDelay)
                        trialTime += (t * timeStep) + motorDelay
                        rt[trialCount] = trialTime
                        trialFinished = True
                        break

                    # Sample the change in RDV from the distribution.
                    RDV += np.random.normal(0, std)

                if trialFinished:
                    break

                # Iterate over the time interval of the current fixation.
                for t in xrange(int(currFixTime // timeStep)):
                    # If the RDV hit one of the barriers, the trial is over.
                    if RDV >= barrier or RDV <= -barrier:
                        if RDV >= barrier:
                            choice[trialCount] = -1
                        elif RDV <= -barrier:
                            choice[trialCount] = 1
                        valLeft[trialCount] = trialCondition[0]
                        valRight[trialCount] = trialCondition[1]
                        fixRDV[trialCount].append(currRDV)
                        fixItem[trialCount].append(currFixItem)
                        fixTime[trialCount].append((t * timeStep) +
                            visualDelay + motorDelay)
                        trialTime += ((t  * timeStep) + visualDelay +
                            motorDelay)
                        rt[trialCount] = trialTime
                        trialFinished = True
                        break

                    # We use a distribution to model changes in RDV
                    # stochastically. The mean of the distribution (the change
                    # most likely to occur) is calculated from the model
                    # parameters and from the values of the two items.
                    if currFixItem == 1:  # Subject is looking left.
                        mean = d * (vLeft - (theta * vRight))
                    elif currFixItem == 2:  # Subject is looking right.
                        mean = d * (-vRight + (theta * vLeft))
                    else:
                         # When not fixating, do we want the RDV to drift?
                        if nonFixDiffusion == True: 
                            mean = d * (vLeft - vRight)
                        else:
                            mean = 0

                    # Sample the change in RDV from the distribution.
                    RDV += np.random.normal(mean, std)

                if trialFinished:
                    break

                # Add previous fixation to this trial's data.
                fixRDV[trialCount].append(currRDV)
                fixItem[trialCount].append(currFixItem)
                fixTime[trialCount].append(((t + 1) * timeStep) + visualDelay)
                trialTime += ((t + 1) * timeStep) + visualDelay

                # Sample and iterate over transition time.
                transitionTime = np.random.choice(distTransition)
                currRDV = RDV
                for t in xrange(int(transitionTime // timeStep)):
                    # If the RDV hit one of the barriers, we abort the trial,
                    # since a trial must end on an item fixation.
                    if RDV >= barrier or RDV <= -barrier:
                        trialFinished = True
                        trialAborted = True
                        break

                    # Sample the change in RDV from the distribution.
                    RDV += np.random.normal(0, std)

                if trialFinished:
                    break

                # Add previous transition to this trial's data.
                fixRDV[trialCount].append(currRDV)
                fixItem[trialCount].append(0)
                fixTime[trialCount].append((t + 1) * timeStep)
                trialTime += (t + 1) * timeStep

                # Sample the next fixation for this trial.
                if currFixItem == 1:
                    currFixItem = 2
                elif currFixItem == 2:
                    currFixItem = 1
                if fixNumber == 2:
                    fixNumber += 1
                    currFixTime = (np.random.choice(distSecondFix[valueDiff]) -
                        visualDelay)
                    continue
                
                # If we have more than two item fixations, we simply sample from
                # the second fixation distribution
                if fixNumber > 2:
                    fixNumber += 1
                    currFixTime = (np.random.choice(distSecondFix[valueDiff]) -
                        visualDelay)
                    continue


            # Move on to the next trial.
            if not trialAborted:
                trial += 1
                trialCount += 1

    simul = collections.namedtuple('Simul', ['rt', 'choice', 'valLeft',
        'valRight', 'fixItem', 'fixTime', 'fixRDV'])
    return simul(rt, choice, valLeft, valRight, fixItem, fixTime, fixRDV)
