#!/usr/bin/python

# dyn_prog_fixations.py
# Author: Gabriela Tavares, gtavares@caltech.edu

from multiprocessing import Pool
from numba import jit
from scipy.stats import norm

import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_data_from_csv():
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
                (df['parcode']==subject), ['rt','choice','dist_left',
                'dist_right']])
            rt[subject][trial] = dataTrial[0,0]
            choice[subject][trial] = dataTrial[0,1]
            leftValue[subject][trial] = np.absolute(
                (np.absolute(dataTrial[0,2])-15)/5)
            rightValue[subject][trial] = np.absolute(
                (np.absolute(dataTrial[0,3])-15)/5)

    # Load fixation data from CSV file.
    # Format: parcode, trial, fix_item, fix_time.
    df = pd.DataFrame.from_csv('fixations.csv', header=0, sep=',',
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

    data = collections.namedtuple('Data', ['rt', 'choice', 'leftValue',
        'rightValue', 'fixItem', 'fixTime'])
    return data(rt, choice, leftValue, rightValue, fixItem, fixTime)


@jit("(f8,f8,f8,f8,f8[:],f8[:],f8,f8,f8)")
def analysis_per_trial(rt, choice, leftValue, rightValue, fixItem, fixTime, d,
    theta, std):
    # Parameters of the grid.
    stateStep = 0.1
    timeStep = 10
    initialBarrierUp = 1
    initialBarrierDown = -1

    # Iterate over the fixations and get the transition time for this
    # trial.
    itemFixTime = 0
    transitionTime = 0
    for fItem, fTime in zip(fixItem, fixTime):
        if fItem == 1 or fItem == 2:
            itemFixTime += int(fTime/timeStep)
        else:
            transitionTime += int(fTime/timeStep)

    # The total time of this trial is given by the sum of all fixations
    # in the trial.
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
    states = np.arange(initialBarrierDown, initialBarrierUp+stateStep,stateStep)
    idx = np.where(np.logical_and(states<0.01, states>-0.01))[0]
    states[idx] = 0

    # Initial probability for all states is zero, except for the zero
    # state, which has initial probability equal to one.
    prStates = np.zeros(states.size)
    idx = np.where(states==0)[0]
    prStates[idx] = 1

    # The probability of crossing each barrier over the time of the
    # trial.
    probUpCrossing = np.zeros(maxTime)
    probDownCrossing = np.zeros(maxTime)

    # Iterate over all fixations in this trial.
    for fItem, fTime in zip(fixItem, fixTime):
        if fItem == 1:  # subject is looking left.
            valueLooking = leftValue
            valueNotLooking = rightValue
        elif fItem == 2:  # subject is looking right.
            valueLooking = rightValue
            valueNotLooking = leftValue
        else:
            continue

        # The mean of the distribution is calculated from the model
        # parameters and from the values of the two items.
        mean = d * (valueLooking - theta*valueNotLooking)

        # Iterate over the time interval of this fixation.
        for t in xrange(0, int(fTime/timeStep)):
            prStatesNew = np.zeros(states.size)

            # Update the probability of the states that remain inside
            # the barriers.
            for s in xrange(0,states.size):
                currState = states[s]
                if (currState > barrierDown[time] and
                    currState < barrierUp[time]):
                    change = (currState * np.ones(states.size)) - states
                    # The probability of being in state B is the sum,
                    # over all states A, of the probability of being in
                    # A at the previous timestep times the probability
                    # of changing from A to B.
                    prStatesNew[s] = (stateStep * np.sum(np.multiply(prStates,
                        norm.pdf(change,mean,std))))

            # Calculate the probabilities of crossing the up barrier and
            # the down barrier. This is given by the sum, over all
            # states A, of the probability of being in A at the previous
            # timestep times the probability of crossing the barrier if
            # A is the previous sstate.
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

            # Update the probabilities of each state and the
            # probabilities of crossing each barrier at this timestep.
            prStates = prStatesNew
            probUpCrossing[time] = tempUpCross
            probDownCrossing[time] = tempDownCross

            time += 1

    # Compute the likelihood contribution of this trial based on the
    # final choice.
    if choice == -1:  # choice was left.
        likelihood = np.log(probUpCrossing[-1])
    elif choice == 1:  # choice was right.
        likelihood = np.log(probDownCrossing[-1])
    return likelihood


def run_analysis(rt, choice, leftValue, rightValue, fixItem, fixTime, d, theta,
    std):
    subjects = rt.keys()
    likelihood = 0

    for subject in subjects:
        print 'Running subject ' + subject + '...'
        trials = rt[subject].keys()
        for trial in trials:
            if trial % 200 == 0:
                print 'Trial ' + str(trial)
            likelihood -= analysis_per_trial(rt[subject][trial],
                choice[subject][trial], leftValue[subject][trial],
                rightValue[subject][trial], fixItem[subject][trial],
                fixTime[subject][trial], d, theta, std)
            
    print 'Likelihood: ' + str(likelihood)
    return likelihood


def run_analysis_wrapper(params):
    return run_analysis(*params)


def main():
    numThreads = 4
    pool = Pool(numThreads)

    data = get_data_from_csv()
    rt = data.rt
    choice = data.choice
    leftValue = data.leftValue
    rightValue = data.rightValue
    fixItem = data.fixItem
    fixTime = data.fixTime

    # Coarse grid search on the parameters of the model.
    print 'Starting coarse grid search...'
    rangeD = [0.0015, 0.002, 0.0025]
    rangeTheta = [0.5, 0.7, 0.9]
    rangeStd = [0.15, 0.2, 0.25]

    numIterations = len(rangeD) * len(rangeTheta) * len(rangeStd)
    models = list()
    list_params = list()
    for d in rangeD:
        for theta in rangeTheta:
            for std in rangeStd:
                models.append((d, theta, std))
                params = (rt, choice, leftValue, rightValue, fixItem, fixTime,
                    d, theta, std)
                list_params.append(params)

    print 'Starting pool of workers...'
    results_coarse = pool.map(run_analysis_wrapper, list_params)
    pool.close()
    pool.join()
    print [r for r in results_coarse]

    # Get optimal parameters.
    max_likelihood_idx = results_coarse.index(max(results_coarse))
    optimD = models[max_likelihood_idx][0]
    optimTheta = models[max_likelihood_idx][1]
    optimStd = models[max_likelihood_idx][2]
    print 'Finished coarse grid search!'
    print 'Optimal d: ' + str(optimD)
    print 'Optimal theta: ' + str(optimTheta)
    print 'Optimal std: ' + str(optimStd)

    # Fine grid search on the parameters of the model.
    print 'Starting fine grid search...'
    rangeD = [optimD-0.00025, optimD, optimD+0.00025]
    rangeTheta = [optimTheta-0.1, optimTheta, optimTheta+0.1]
    rangeStd = [optimStd-0.025, optimStd, optimStd+0.025]

    numIterations = len(rangeD) * len(rangeTheta) * len(rangeStd)
    models = list()
    list_params = list()
    for d in rangeD:
        for theta in rangeTheta:
            for std in rangeStd:
                models.append((d, theta, std))
                params = (rt, choice, leftValue, rightValue, fixItem, fixTime,
                    d, theta, std)
                list_params.append(params)

    print 'Starting pool of workers...'
    results_fine = pool.map(run_analysis_wrapper, list_params)
    pool.close()
    pool.join()
    print [r for r in results_fine]

    # Get optimal parameters.
    max_likelihood_idx = results_fine.index(max(results_fine))
    optimD = models[max_likelihood_idx][0]
    optimTheta = models[max_likelihood_idx][1]
    optimStd = models[max_likelihood_idx][2]
    print 'Finished fine grid search!'
    print 'Optimal d: ' + str(optimD)
    print 'Optimal theta: ' + str(optimTheta)
    print 'Optimal std: ' + str(optimStd)


if __name__ == '__main__':
    main()