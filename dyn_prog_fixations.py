#!/usr/bin/python

# dyn_prog_fixations.py
# Author: Gabriela Tavares, gtavares@caltech.edu

from scipy.stats import norm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Load experimental data from CSV file.
# Format: parcode, trial, rt, choice, dist_left, dist_right.
df = pd.DataFrame.from_csv('expdata.csv', header=0, sep=',', index_col=None)
subjects = df.parcode.unique()

rt = dict()
choice = dict()
leftValue = dict()
rightValue = dict()

for subject in subjects:
    rt[subject] = dict()
    choice[subject] = dict()
    leftValue[subject] = dict()
    rightValue[subject] = dict()
    dataSubject = np.array(df.loc[df['parcode']==subject,
        ['trial','rt','choice','dist_left','dist_right']])
    trials = np.unique(dataSubject[:,0]).tolist()
    for trial in trials:
        dataTrial = np.array(df.loc[(df['trial']==trial) &
            (df['parcode']==subject), ['rt','choice','dist_left','dist_right']])
        rt[subject][trial] = dataTrial[0,0]
        choice[subject][trial] = dataTrial[0,1]
        leftValue[subject][trial] = np.absolute(
            (np.absolute(dataTrial[0,2])-15)/5)
        rightValue[subject][trial] = np.absolute(
            (np.absolute(dataTrial[0,3])-15)/5)

# Load fixation data from CSV file.
# Format: parcode, trial, fix_item, fix_time.
df = pd.DataFrame.from_csv('fixations.csv', header=0, sep=',', index_col=None)
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

# Parameters of the model
d = 0.002
theta = 0.7
std = 0.25

# Parameters of the grid.
stateStep = 0.1
timeStep = 10
initialBarrierUp = 1
initialBarrierDown = -1

subjects = rt.keys()
likelihood = 0

for subject in subjects:
    print 'Running subject ' + subject + '...'
    trials = rt[subject].keys()
    for trial in trials:
        if trial % 100 == 0:
            print 'Trial ' + str(trial)

        # Iterate over the fixations and get the transition time for this trial.
        itemFixTime = 0
        transitionTime = 0
        for fItem, fTime in zip(fixItem[subject][trial],
            fixTime[subject][trial]):
            if fItem == 1 or fItem == 2:
                itemFixTime += int(fTime/timeStep)
            else:
                transitionTime += int(fTime/timeStep)

        # The total time of this trial is given by the sum of all fixations in
        # the trial.
        maxTime = itemFixTime + transitionTime

        # We start couting the trial time at the end of the transition time.
        time = transitionTime

        # The values of the barriers can change over time.
        decay = 0  # decay = 0 means barriers are constant.
        barrierUp = initialBarrierUp * np.ones(maxTime)
        barrierDown = initialBarrierDown * np.ones(maxTime)
        for t in xrange(0,maxTime):
            barrierUp[t] = initialBarrierUp / (1+decay*(t+1))
            barrierDown[t] = initialBarrierDown / (1+decay*(t+1))

        # The vertical axis is divided into states.
        states = np.arange(initialBarrierDown, initialBarrierUp+stateStep,
            stateStep)
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

        # Iterate over all fixations in this trial.
        for fItem, fTime in zip(fixItem[subject][trial],
            fixTime[subject][trial]):
            if fItem == 1:  # subject is looking left.
                valueLooking = leftValue[subject][trial]
                valueNotLooking = rightValue[subject][trial]
            elif fItem == 2:  # subject is looking right.
                valueLooking = rightValue[subject][trial]
                valueNotLooking = leftValue[subject][trial]
            else:
                continue

            # The mean of the distribution is calculated from the model params
            # and from the values of the two items.
            mean = d * (valueLooking - theta*valueNotLooking)

            # Iterate over the time interval of this fixation.
            for t in xrange(0, int(fTime/timeStep)):
                prStatesNew = np.zeros(states.size)

                # Update the probability of the states that remain inside the
                # barriers.
                for s in xrange(0,states.size):
                    currState = states[s]
                    if (currState > barrierDown[time] and
                        currState < barrierUp[time]):
                        change = (currState * np.ones(states.size)) - states
                         # The probability of being in state B is the sum, over
                         # all states A, of the probability of being in A at the
                         # previous timestep times the probability of changing
                         # from A to B.
                        prStatesNew[s] = (stateStep *
                            np.sum(np.multiply(prStates,
                            norm.pdf(change,mean,std))))

                # Calculate the probabilities of crossing the up barrier and the
                # down barrier. This is given by the sum, over all states A, of
                # the probability of being in A at the previous timestep times
                # the probability of crossing the barrier if A is the previous
                # state.
                changeUp = (barrierUp[time] * np.ones(states.size)) - states
                tempUpCross = np.sum(np.multiply(prStates,
                    (1 - norm.cdf(changeUp,mean,std))))
                changeDown = (barrierDown[time] * np.ones(states.size)) - states
                tempDownCross = np.sum(np.multiply(prStates,
                    (norm.cdf(changeDown,mean,std))))

                # Renormalize to cope with numerical approximation.
                sumIn = np.sum(prStates)
                sumCurrent = np.sum(prStatesNew) + tempUpCross + tempDownCross
                prStatesNew = prStatesNew * sumIn/sumCurrent
                tempUpCross = tempUpCross * sumIn/sumCurrent
                tempDownCross = tempDownCross * sumIn/sumCurrent

                # Update the probabilities of each state and the probabilities
                # of crossing each barrier at this timestep.
                prStates = prStatesNew
                probUpCrossing[time] = tempUpCross
                probDownCrossing[time] = tempDownCross

                time += 1

        # Compute the likelihood contribution of this trial based on the final
        # choice.
        try:
            if choice[subject][trial] == -1:  # choice was left.
                likelihood -= np.log(probUpCrossing[-1])
            elif choice[subject][trial] == 1:  # choice was right.
                likelihood -= np.log(probDownCrossing[-1])
        except MemoryError:
            print 'Memory error!'
        except OverflowError:
            print 'Overflow error!'

print 'Likelihood: ' + str(likelihood)
